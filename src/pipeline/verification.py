"""Part A1: Document Verification Stage (paper sections 4.2.1 to 4.2.4).

Executes after ingestion and before chunking.
Validates each document across four dimensions:
  1. Quality: checks for empty text, garbled extraction, and prompt injection attempts
     using shared patterns from src/pipeline/guardrails.py.
  2. Completeness: ensures front matter closes, mandatory metadata (source, title,
     positive int page_count, ISO date, doc_version) is present, and body passes
     truncation & length-per-page sanity checks.
  3. Recency: groups by doc_family, parses ISO dates (falling back to version tuples),
     and detects when a document is superseded by a newer member of the same family.
  4. Uniqueness: flags exact duplicates by SHA-256 body hash and near-duplicates via
     cosine similarity over whole-body embeddings.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any

from .config import get_embedding_version, get_results_dir
from .embed import DEFAULT_EMBEDDING_VERSION, cosine_similarity, embed_text
from .guardrails import scan_text_for_injection, scan_text_for_truncation

MANDATORY_METADATA = ["source", "title", "page_count", "doc_version", "date"]
NEAR_DUPLICATE_THRESHOLD = 0.90


@dataclass
class CheckResult:
    check_name: str
    passed: bool
    reason: str


@dataclass
class DocumentVerificationRecord:
    filename: str
    source: str
    doc_family: str
    doc_version: str
    date: str
    passed_all: bool
    action: str  # "accepted" | "quarantined" | "archived"
    checks: dict[str, CheckResult]
    quarantined: bool
    quarantine_path: str | None = None
    quarantine_reason: str | None = None
    max_similarity: float = 0.0
    most_similar_doc: str | None = None


@dataclass
class VerificationReport:
    timestamp: str
    batch_label: str
    embedding_version: str
    total_scanned: int
    passed_count: int
    quarantined_count: int
    archived_count: int
    per_check_summary: dict[str, dict[str, int]]
    records: list[DocumentVerificationRecord] = field(default_factory=list)
    report_path: str | None = None


def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-like markdown front matter between --- markers."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {"_front_matter_closed": False}, content

    meta: dict[str, Any] = {}
    body_start_idx = 0
    closed = False
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start_idx = idx + 1
            closed = True
            break
        key, sep, val = line.partition(":")
        if sep:
            k = key.strip()
            v = val.strip()
            if v.lower() == "true":
                meta[k] = True
            elif v.lower() == "false":
                meta[k] = False
            elif re.fullmatch(r"-?\d+", v):
                meta[k] = int(v)
            else:
                meta[k] = v

    meta["_front_matter_closed"] = closed
    body_text = "\n".join(lines[body_start_idx:]).strip() if closed else ""
    return meta, body_text


def derive_doc_family(meta: dict[str, Any], filename: str = "") -> str:
    """Return explicit doc_family or derive it by stripping year/version tokens."""
    if meta.get("doc_family"):
        return str(meta["doc_family"]).strip().lower()
    raw = str(meta.get("source") or filename or "unknown")
    stem = Path(raw).stem.lower()
    stem = re.sub(r"[-_]?20\d{2}(?:[-_]\d{2})?", "", stem)
    stem = re.sub(r"[-_]?v\d+(?:\.\d+)*", "", stem)
    stem = re.sub(r"[-_]+$", "", stem)
    return stem or "unknown"


def _parse_date(meta: dict[str, Any]) -> datetime.date | None:
    raw = meta.get("date")
    if not raw:
        return None
    try:
        return datetime.date.fromisoformat(str(raw).strip())
    except ValueError:
        return None


def _parse_version_tuple(ver_str: str) -> tuple[int, ...]:
    nums = [int(x) for x in re.findall(r"\d+", str(ver_str))]
    return tuple(nums) if nums else (0,)


def _is_newer_than(other_meta: dict[str, Any], self_meta: dict[str, Any]) -> bool:
    other_date = _parse_date(other_meta)
    self_date = _parse_date(self_meta)
    if other_date is not None and self_date is not None:
        return other_date > self_date
    return _parse_version_tuple(str(other_meta.get("doc_version", ""))) > _parse_version_tuple(
        str(self_meta.get("doc_version", ""))
    )


def check_quality(meta: dict[str, Any], body: str) -> CheckResult:
    """Check for empty, garbled, truncated-error, or prompt-injected text."""
    if len(body.strip()) < 50:
        return CheckResult("quality", False, "Document body is empty or too short (<50 characters)")

    trunc_marker = scan_text_for_truncation(body)
    if trunc_marker or "failed_text_encoding" in body.lower():
        return CheckResult("quality", False, "Extraction failure or truncation marker detected in body")

    non_printable = sum(1 for c in body if ord(c) < 32 and c not in "\n\r\t")
    if non_printable > 5 or "\x00" in body or "\\x00" in body:
        return CheckResult("quality", False, "High ratio of non-printable or null-byte sequences (garbled text)")

    inj = scan_text_for_injection(body)
    if inj:
        return CheckResult("quality", False, f"Prompt injection attempt detected: '{inj}'")

    return CheckResult("quality", True, "Quality check passed (clean, readable text)")


def check_completeness(meta: dict[str, Any], body: str) -> CheckResult:
    """Ensure front matter closes, required metadata parses, and body is complete."""
    if meta.get("_front_matter_closed") is False:
        return CheckResult("completeness", False, "YAML front matter is missing or unclosed")

    missing = []
    for k in MANDATORY_METADATA:
        if k not in meta or str(meta[k]).strip() == "":
            missing.append(k)
    if missing:
        return CheckResult("completeness", False, f"Missing required metadata fields: {', '.join(missing)}")

    page_count = meta.get("page_count")
    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count <= 0:
        return CheckResult("completeness", False, f"Invalid page_count '{page_count}': must be a positive integer")

    if _parse_date(meta) is None:
        return CheckResult("completeness", False, f"Invalid ISO date '{meta.get('date')}' (expected YYYY-MM-DD)")

    if body.endswith("...") or body.endswith("[truncated]") or len(body.splitlines()) < 3:
        return CheckResult("completeness", False, "Document body appears abruptly truncated (<3 lines or trailing ellipsis)")

    # Length-per-page sanity check for condensed policy documents
    chars_per_page = len(body.strip()) / float(page_count)
    if chars_per_page < 10.0 and len(body.strip()) < 300:
        return CheckResult(
            "completeness",
            False,
            f"Body length ({len(body.strip())} chars) too sparse for declared page_count={page_count}",
        )

    return CheckResult("completeness", True, "Completeness check passed (metadata valid and structure intact)")


def check_recency(meta: dict[str, Any], known_docs: list[dict[str, Any]], filename: str = "") -> CheckResult:
    """Detect if document is explicitly superseded or superseded by a newer family member."""
    current_version = str(meta.get("doc_version", ""))
    if meta.get("superseded", False) is True:
        return CheckResult("recency", False, f"Document explicitly marked superseded (version {current_version})")

    my_family = derive_doc_family(meta, filename)
    my_source = str(meta.get("source", filename))

    for other in known_docs:
        other_source = str(other.get("source", ""))
        if other_source == my_source and str(other.get("doc_version", "")) == current_version:
            continue
        if derive_doc_family(other) == my_family and _is_newer_than(other, meta):
            return CheckResult(
                "recency",
                False,
                f"superseded by {other_source} dated {other.get('date', other.get('doc_version'))}",
            )

    return CheckResult("recency", True, f"Document is current for family '{my_family}' (version {current_version})")


def check_uniqueness(
    doc_path: Path,
    body: str,
    body_hash: str,
    embedding: list[float],
    indexed_hashes: dict[str, str],
    indexed_embeddings: dict[str, list[float]],
) -> tuple[CheckResult, float, str | None]:
    """Check for exact SHA-256 duplicates and near-duplicates via whole-body cosine similarity."""
    for existing_file, ex_hash in indexed_hashes.items():
        if existing_file != doc_path.name and ex_hash == body_hash:
            return (
                CheckResult("uniqueness", False, f"Exact duplicate of existing document: {existing_file}"),
                1.0,
                existing_file,
            )

    max_sim = 0.0
    best_doc: str | None = None
    for existing_file, ex_emb in indexed_embeddings.items():
        if existing_file != doc_path.name:
            sim = cosine_similarity(embedding, ex_emb)
            if sim > max_sim:
                max_sim = sim
                best_doc = existing_file

    if max_sim >= NEAR_DUPLICATE_THRESHOLD and best_doc is not None:
        return (
            CheckResult(
                "uniqueness",
                False,
                f"Near-duplicate of {best_doc} (cosine similarity: {max_sim:.4f} >= {NEAR_DUPLICATE_THRESHOLD})",
            ),
            round(max_sim, 4),
            best_doc,
        )

    return (
        CheckResult(
            "uniqueness",
            True,
            f"Document is unique (max similarity {max_sim:.4f} < {NEAR_DUPLICATE_THRESHOLD})",
        ),
        round(max_sim, 4),
        best_doc,
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def run_verification(
    corpus_dir: str | Path,
    quarantine_dir: str | Path,
    output_dir: str | Path | None = None,
    reference_corpus_dir: str | Path | None = None,
    archive_dir: str | Path | None = None,
    batch_label: str = "batchA",
    embedding_version: str | None = None,
) -> VerificationReport:
    """Run 4-way verification on corpus_dir against optional reference_corpus_dir.

    - Files failing Quality, Completeness, or Uniqueness are moved to quarantine_dir.
    - Files failing only Recency are archived (if archive_dir is provided) or tagged archived.
    - Reference documents superseded by a newly verified document are also recorded with
      action="archived" in the verification report.
    """
    corpus_path = Path(corpus_dir)
    quarantine_path = Path(quarantine_dir)
    quarantine_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) if output_dir else get_results_dir()
    out_path.mkdir(parents=True, exist_ok=True)
    emb_ver = embedding_version or get_embedding_version() or DEFAULT_EMBEDDING_VERSION

    indexed_hashes: dict[str, str] = {}
    indexed_embeddings: dict[str, list[float]] = {}
    ref_docs: list[dict[str, Any]] = []

    if reference_corpus_dir:
        ref_path = Path(reference_corpus_dir)
        if ref_path.exists():
            for ref_file in sorted(ref_path.glob("*.md")):
                if ref_file.name.startswith("."):
                    continue
                r_text = ref_file.read_text(encoding="utf-8", errors="ignore")
                r_meta, r_body = parse_front_matter(r_text)
                r_meta["_filename"] = ref_file.name
                r_meta["_path"] = ref_file
                indexed_hashes[ref_file.name] = hashlib.sha256(r_body.encode("utf-8")).hexdigest()
                indexed_embeddings[ref_file.name] = embed_text(r_body, version=emb_ver)
                ref_docs.append(r_meta)

    files = sorted([f for f in corpus_path.glob("*.md") if not f.name.startswith(".")])

    docs_data: list[dict[str, Any]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_front_matter(text)
        b_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        emb = embed_text(body, version=emb_ver)
        docs_data.append({
            "path": f,
            "meta": meta,
            "body": body,
            "hash": b_hash,
            "embedding": emb,
        })

    # Pre-check which batch docs pass hard checks (Quality + Completeness) so defective docs
    # like doc_b04 cannot falsely supersede valid documents in recency comparison
    valid_batch_metas: list[dict[str, Any]] = []
    for d in docs_data:
        if check_quality(d["meta"], d["body"]).passed and check_completeness(d["meta"], d["body"]).passed:
            valid_batch_metas.append(d["meta"])

    all_valid_metas = ref_docs + valid_batch_metas

    records: list[DocumentVerificationRecord] = []
    quarantined_count = 0
    archived_count = 0
    passed_count = 0

    per_check_summary: dict[str, dict[str, int]] = {
        "quality": {"passed": 0, "failed": 0},
        "completeness": {"passed": 0, "failed": 0},
        "recency": {"passed": 0, "failed": 0},
        "uniqueness": {"passed": 0, "failed": 0},
    }

    timestamp = datetime.datetime.now(timezone.utc).isoformat()

    # Check incoming / batch files
    newly_accepted_metas: list[dict[str, Any]] = []
    for item in docs_data:
        f_path: Path = item["path"]
        meta = item["meta"]
        body = item["body"]
        b_hash = item["hash"]
        emb = item["embedding"]

        c_qual = check_quality(meta, body)
        c_comp = check_completeness(meta, body)
        c_rec = check_recency(meta, all_valid_metas, filename=f_path.name)
        c_uniq, max_sim, sim_doc = check_uniqueness(
            f_path, body, b_hash, emb, indexed_hashes, indexed_embeddings
        )

        checks_dict = {
            "quality": c_qual,
            "completeness": c_comp,
            "recency": c_rec,
            "uniqueness": c_uniq,
        }
        for cname, cres in checks_dict.items():
            per_check_summary[cname]["passed" if cres.passed else "failed"] += 1

        is_hard_failure = not (c_qual.passed and c_comp.passed and c_uniq.passed)
        passed_all = c_qual.passed and c_comp.passed and c_rec.passed and c_uniq.passed

        quarantined = False
        q_path_str = None
        q_reason = None
        action = "accepted"

        if is_hard_failure:
            failing_checks = [c for c in (c_qual, c_comp, c_uniq) if not c.passed]
            q_reason = "; ".join(f"[{c.check_name.upper()} FAIL] {c.reason}" for c in failing_checks)
            target_file = quarantine_path / f_path.name
            shutil.move(str(f_path), str(target_file))
            quarantined = True
            quarantined_count += 1
            q_path_str = str(target_file)
            action = "quarantined"
        elif not c_rec.passed:
            action = "archived"
            archived_count += 1
            if archive_dir:
                arch_p = Path(archive_dir)
                arch_p.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f_path), str(arch_p / f_path.name))
        else:
            indexed_hashes[f_path.name] = b_hash
            indexed_embeddings[f_path.name] = emb
            passed_count += 1
            newly_accepted_metas.append(meta)

        records.append(
            DocumentVerificationRecord(
                filename=f_path.name,
                source=str(meta.get("source", f_path.name)),
                doc_family=derive_doc_family(meta, f_path.name),
                doc_version=str(meta.get("doc_version", "unknown")),
                date=str(meta.get("date", "unknown")),
                passed_all=passed_all,
                action=action,
                checks=checks_dict,
                quarantined=quarantined,
                quarantine_path=q_path_str,
                quarantine_reason=q_reason,
                max_similarity=max_sim,
                most_similar_doc=sim_doc,
            )
        )

    # Also check if any reference corpus document was superseded by a newly accepted batch document
    if newly_accepted_metas and ref_docs:
        for ref_meta in ref_docs:
            ref_recency = check_recency(ref_meta, newly_accepted_metas, filename=ref_meta["_filename"])
            if not ref_recency.passed and not ref_meta.get("superseded", False):
                per_check_summary["recency"]["failed"] += 1
                archived_count += 1
                records.append(
                    DocumentVerificationRecord(
                        filename=ref_meta["_filename"],
                        source=str(ref_meta.get("source", ref_meta["_filename"])),
                        doc_family=derive_doc_family(ref_meta, ref_meta["_filename"]),
                        doc_version=str(ref_meta.get("doc_version", "unknown")),
                        date=str(ref_meta.get("date", "unknown")),
                        passed_all=False,
                        action="archived",
                        checks={"recency": ref_recency},
                        quarantined=False,
                        quarantine_reason=ref_recency.reason,
                    )
                )

    report_filename = f"verification_{batch_label}_{int(time.time())}.json"
    report_file = out_path / report_filename
    report = VerificationReport(
        timestamp=timestamp,
        batch_label=batch_label,
        embedding_version=emb_ver,
        total_scanned=len(files),
        passed_count=passed_count,
        quarantined_count=quarantined_count,
        archived_count=archived_count,
        per_check_summary=per_check_summary,
        records=records,
        report_path=str(report_file),
    )

    serialized = {
        "timestamp": report.timestamp,
        "batch_label": report.batch_label,
        "embedding_version": report.embedding_version,
        "total_scanned": report.total_scanned,
        "passed_count": report.passed_count,
        "quarantined_count": report.quarantined_count,
        "archived_count": report.archived_count,
        "per_check_summary": report.per_check_summary,
        "records": [
            {
                "filename": r.filename,
                "source": r.source,
                "doc_family": r.doc_family,
                "doc_version": r.doc_version,
                "date": r.date,
                "passed_all": r.passed_all,
                "action": r.action,
                "quarantined": r.quarantined,
                "quarantine_reason": r.quarantine_reason,
                "max_similarity": r.max_similarity,
                "most_similar_doc": r.most_similar_doc,
                "checks": {
                    k: {"passed": v.passed, "reason": v.reason} for k, v in r.checks.items()
                },
            }
            for r in report.records
        ],
    }
    _atomic_write_json(report_file, serialized)
    return report
