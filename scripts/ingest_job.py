"""Production Sole-Writer Ingestion & Verification Pipeline Job (Part A5 & F2).

Scans LAKE_DIR/incoming/:
  1. If incoming files exist, runs 4-way verification against LAKE_DIR/raw/.
  2. Quarantined files are moved to LAKE_DIR/quarantine/.
  3. Superseded files (including older raw/ family members) are archived to LAKE_DIR/archive/.
  4. Passing files move into LAKE_DIR/raw/.
  5. Runs incremental ingestion into INDEX_DIR/${COLLECTION_NAME}.json + manifest.json.
  6. Writes verification_<batch>_<ts>.json and ingestion_<ts>.json to RESULTS_DIR.

Exits 0 even when incoming documents are quarantined; exits non-zero only on
unexpected infrastructure errors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import (
    get_archive_dir,
    get_chunk_overlap,
    get_chunk_size,
    get_collection_name,
    get_embedding_version,
    get_incoming_dir,
    get_index_dir,
    get_quarantine_dir,
    get_raw_dir,
    get_results_dir,
)
from src.pipeline.indexer import IncrementalIndexer
from src.pipeline.verification import run_verification


def run_ingest_job(
    batch_label: str = "incoming",
    collection_name: str | None = None,
    embedding_version: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    incoming_dir = get_incoming_dir()
    raw_dir = get_raw_dir()
    quarantine_dir = get_quarantine_dir()
    archive_dir = get_archive_dir()
    index_dir = get_index_dir()
    results_dir = get_results_dir()

    for d in (incoming_dir, raw_dir, quarantine_dir, archive_dir, index_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    col = collection_name or get_collection_name()
    emb_ver = embedding_version or get_embedding_version()
    c_size = chunk_size if chunk_size is not None else get_chunk_size()
    c_overlap = chunk_overlap if chunk_overlap is not None else get_chunk_overlap()

    indexer = IncrementalIndexer(
        corpus_dir=raw_dir,
        archive_dir=archive_dir,
        data_dir=index_dir,
        collection_name=col,
        embedding_version=emb_ver,
    )

    incoming_files = sorted(f for f in incoming_dir.glob("*.md") if not f.name.startswith("."))
    verification_report = None

    if incoming_files:
        print(f"[IngestJob] Found {len(incoming_files)} file(s) in {incoming_dir}. Running verification...")
        verification_report = run_verification(
            corpus_dir=incoming_dir,
            quarantine_dir=quarantine_dir,
            output_dir=results_dir,
            reference_corpus_dir=raw_dir,
            archive_dir=archive_dir,
            batch_label=batch_label,
            embedding_version=emb_ver,
        )
        print(
            f"[IngestJob] Verification summary: scanned={verification_report.total_scanned}, "
            f"passed={verification_report.passed_count}, quarantined={verification_report.quarantined_count}, "
            f"archived={verification_report.archived_count}"
        )

        # Archive any raw/ documents superseded by newly verified documents
        for rec in verification_report.records:
            if rec.action == "archived" and (raw_dir / rec.filename).exists():
                indexer.archive_superseded(rec.filename)

        # Move remaining passing incoming files into raw_dir
        for passer in sorted(incoming_dir.glob("*.md")):
            target = raw_dir / passer.name
            shutil.move(str(passer), str(target))
            print(f"[IngestJob] Promoted verified file to lake raw/: {passer.name}")
    else:
        print(f"[IngestJob] No new files in {incoming_dir}; running incremental sync on {raw_dir}...")

    stats = indexer.run_ingestion(chunk_size=c_size, chunk_overlap=c_overlap)

    ts = int(time.time())
    ing_file = results_dir / f"ingestion_{col}_{ts}.json"
    tmp = ing_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    os.replace(tmp, ing_file)

    print(
        f"[IngestJob] Collection '{col}' ({emb_ver}): "
        f"added={stats['added']}, modified={stats['modified']}, unchanged={stats['unchanged']}, "
        f"deleted={stats['deleted']}, chunks_embedded={stats['chunks_embedded']}, "
        f"active_chunks={stats['active_chunks_in_store']}"
    )
    print(f"[IngestJob] Stats written to {ing_file}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGOps Sole-Writer Verification + Ingestion Job")
    parser.add_argument("--batch-label", default=os.environ.get("BATCH_LABEL", "incoming"))
    parser.add_argument("--collection", default=None)
    parser.add_argument("--embedding-version", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    try:
        run_ingest_job(
            batch_label=args.batch_label,
            collection_name=args.collection,
            embedding_version=args.embedding_version,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as exc:
        print(f"[IngestJob] FATAL infrastructure error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
