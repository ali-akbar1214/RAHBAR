"""Part C: Change Experiment & Automated Release Gate (C1 - C4, F5).

Evaluates candidate pipeline configurations in isolated temporary directories or
dedicated collection files (never overwriting the live production vector store),
compares against committed eval/baseline.json, enforces all thresholds from
eval/gate_config.yaml (with generic env overrides), writes
eval/results/gate_<candidate>_<ts>.json, and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
import time
from datetime import timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import print_summary, run_evaluation_suite
from src.pipeline.config import (
    PROJECT_ROOT,
    get_raw_dir,
    get_results_dir,
    load_gate_config,
)
from src.pipeline.indexer import IncrementalIndexer


def check_release_gate(
    candidate_res: dict[str, Any],
    candidate_name: str,
    baseline_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate candidate_res against baseline.json and gate_config.yaml."""
    cfg = load_gate_config()
    b_path = baseline_path or (PROJECT_ROOT / "eval" / "baseline.json")
    if not b_path.exists():
        raise FileNotFoundError(f"Baseline file missing: {b_path}. Run eval/run_eval.py --save-baseline first.")
    baseline = json.loads(b_path.read_text(encoding="utf-8"))

    out_dir = output_dir or get_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    b_r5 = float(baseline["component_level"]["clean"]["recall_at_5"])
    c_r5 = float(candidate_res["component_level"]["clean"]["recall_at_5"])
    recall_drop = round(b_r5 - c_r5, 4)

    b_abs = float(baseline["end_to_end"]["abstention_correctness"])
    c_abs = float(candidate_res["end_to_end"]["abstention_correctness"])
    abstention_drop = round(b_abs - c_abs, 4)

    b_false_abs = float(baseline["end_to_end"]["false_abstention_rate"])
    c_false_abs = float(candidate_res["end_to_end"]["false_abstention_rate"])

    c_faith = float(candidate_res["end_to_end"]["faithfulness"])
    c_cit = float(candidate_res["end_to_end"]["citation_validity"])
    c_p95 = float(candidate_res["end_to_end"]["latency_p95_ms"])
    c_det = float(candidate_res["module_level"]["determinism_max_diff"])
    c_fb = int(candidate_res["end_to_end"]["used_fallback_count"])

    max_fa_thresh = float(cfg.get("max_false_abstention_rate", 0.25))

    checks = [
        {
            "rule": "max_recall_at_5_drop",
            "threshold": f"<= {cfg['max_recall_at_5_drop']}",
            "measured": recall_drop,
            "baseline": b_r5,
            "candidate": c_r5,
            "passed": recall_drop <= float(cfg["max_recall_at_5_drop"]),
        },
        {
            "rule": "min_recall_at_5",
            "threshold": f">= {cfg['min_recall_at_5']}",
            "measured": c_r5,
            "passed": c_r5 >= float(cfg["min_recall_at_5"]),
        },
        {
            "rule": "max_abstention_drop",
            "threshold": f"<= {cfg['max_abstention_drop']}",
            "measured": abstention_drop,
            "baseline": b_abs,
            "candidate": c_abs,
            "passed": abstention_drop <= float(cfg["max_abstention_drop"]),
        },
        {
            "rule": "max_false_abstention_rate",
            "threshold": f"<= {max_fa_thresh}",
            "measured": c_false_abs,
            "baseline": b_false_abs,
            "candidate": c_false_abs,
            "passed": c_false_abs <= max_fa_thresh,
        },
        {
            "rule": "min_faithfulness",
            "threshold": f">= {cfg['min_faithfulness']}",
            "measured": c_faith,
            "passed": c_faith >= float(cfg["min_faithfulness"]),
        },
        {
            "rule": "min_citation_validity",
            "threshold": f">= {cfg['min_citation_validity']}",
            "measured": c_cit,
            "passed": c_cit >= float(cfg["min_citation_validity"]),
        },
        {
            "rule": "max_latency_p95_ms",
            "threshold": f"<= {cfg['max_latency_p95_ms']}",
            "measured": c_p95,
            "passed": c_p95 <= float(cfg["max_latency_p95_ms"]),
        },
        {
            "rule": "max_determinism_diff",
            "threshold": f"<= {cfg['max_determinism_diff']}",
            "measured": c_det,
            "passed": c_det <= float(cfg["max_determinism_diff"]),
        },
        {
            "rule": "max_fallback_count",
            "threshold": f"<= {cfg['max_fallback_count']}",
            "measured": c_fb,
            "passed": c_fb <= int(cfg["max_fallback_count"]),
        },
    ]

    all_passed = all(ch["passed"] for ch in checks)
    exit_code = 0 if all_passed else 1

    gate_report = {
        "candidate": candidate_name,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "collection_name": candidate_res.get("collection_name"),
        "embedding_version": candidate_res.get("embedding_version"),
        "verdict": "APPROVED / PASS" if all_passed else "REJECTED / FAIL",
        "exit_code": exit_code,
        "checks": checks,
    }

    gate_file = out_dir / f"gate_{candidate_name}_{int(time.time())}.json"
    tmp = gate_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
    os.replace(tmp, gate_file)

    print("\n" + "=" * 82)
    print(f"  RELEASE GATE VERDICT [{candidate_name}]: {gate_report['verdict']} (exit={exit_code})")
    print("=" * 82)
    print(f"  {'Rule':<24} {'Threshold':<14} {'Measured':<14} {'Verdict'}")
    print("  " + "-" * 64)
    for ch in checks:
        v_str = "PASS" if ch["passed"] else "FAIL"
        print(f"  {ch['rule']:<24} {ch['threshold']:<14} {str(ch['measured']):<14} {v_str}")
    print(f"  Gate report written to: {gate_file}")
    print("=" * 82)

    return all_passed, gate_report


def evaluate_candidate(
    candidate_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_version: str,
    require_llm: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Build candidate index in an isolated TemporaryDirectory and evaluate against gate."""
    with tempfile.TemporaryDirectory(prefix=f"ragops_cand_{candidate_name}_") as tmp:
        tmp_path = Path(tmp)
        col_name = f"cand-{candidate_name}"
        indexer = IncrementalIndexer(
            corpus_dir=get_raw_dir(),
            archive_dir=tmp_path / "archive",
            data_dir=tmp_path,
            collection_name=col_name,
            embedding_version=embedding_version,
        )
        stats = indexer.run_ingestion(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(
            f"[Candidate '{candidate_name}'] Built in temp dir: "
            f"embedding={embedding_version}, chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"chunks={stats['active_chunks_in_store']}"
        )

        res = run_evaluation_suite(
            golden_path="eval/golden.jsonl",
            vector_store_path=indexer.index_path,
            collection_name=col_name,
            embedding_version=embedding_version,
            label=candidate_name,
            require_llm=require_llm,
        )
        print_summary(res)
        return check_release_gate(res, candidate_name=candidate_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part C: Candidate Experiment & Release Gate")
    parser.add_argument(
        "--candidate",
        choices=["chunk", "embed", "fail", "all"],
        default="all",
        help="chunk = candidateA-chunk (500/100, v1); embed = candidateA-embed (320/80, v2); fail = candidateB-bad (40/5, v1)",
    )
    parser.add_argument("--require-llm", action="store_true", help="Fail if Groq LLM fallback triggers")
    args = parser.parse_args()

    candidates_run: list[tuple[str, bool]] = []
    if args.candidate in {"chunk", "all"}:
        ok, _ = evaluate_candidate(
            candidate_name="candidateA-chunk",
            chunk_size=500,
            chunk_overlap=100,
            embedding_version="lexical-proj-v1",
            require_llm=args.require_llm,
        )
        candidates_run.append(("candidateA-chunk", ok))

    if args.candidate in {"embed", "all"}:
        ok, _ = evaluate_candidate(
            candidate_name="candidateA-embed",
            chunk_size=320,
            chunk_overlap=80,
            embedding_version="lexical-proj-v2",
            require_llm=args.require_llm,
        )
        candidates_run.append(("candidateA-embed", ok))

    if args.candidate in {"fail", "all"}:
        ok, _ = evaluate_candidate(
            candidate_name="candidateB-bad",
            chunk_size=40,
            chunk_overlap=5,
            embedding_version="lexical-proj-v1",
            require_llm=args.require_llm,
        )
        candidates_run.append(("candidateB-bad", ok))

    if args.candidate == "fail" and not candidates_run[-1][1]:
        sys.exit(1)
    if args.candidate in {"chunk", "embed"} and not candidates_run[-1][1]:
        sys.exit(1)


if __name__ == "__main__":
    main()
