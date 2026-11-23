"""Part E3: Data Drift Simulation & Versioned Golden Set Transition.

Simulates ingesting Batch B (including academic-rules-2027.md, ai-ethics, hostel)
in an isolated TemporaryDirectory, evaluates:
  1. Baseline golden.jsonl (v1) on drifted index -> shows recall drop because
     academic-rules-2026.md was archived and new docs add distractors.
  2. Updated golden_v2.jsonl on drifted index -> shows recovery + new Batch B coverage.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import print_summary, run_evaluation_suite
from src.pipeline.config import PROJECT_ROOT, get_raw_dir
from src.pipeline.indexer import IncrementalIndexer
from src.pipeline.verification import run_verification


def main() -> None:
    print("=" * 78)
    print("PART E3: DATA DRIFT AFTER BATCH B INGESTION & GOLDEN_V2 UPDATE")
    print("=" * 78)

    with tempfile.TemporaryDirectory(prefix="ragops_e3_") as tmp:
        tmp_path = Path(tmp)
        tmp_raw = tmp_path / "raw"
        tmp_b = tmp_path / "batch_b"
        tmp_q = tmp_path / "quarantine"
        tmp_a = tmp_path / "archive"
        tmp_data = tmp_path / "data"

        shutil.copytree(get_raw_dir(), tmp_raw)
        shutil.copytree(PROJECT_ROOT / "corpus" / "batch_b", tmp_b)

        indexer = IncrementalIndexer(
            corpus_dir=tmp_raw,
            archive_dir=tmp_a,
            data_dir=tmp_data,
            collection_name="fast-policies-drifted",
            embedding_version="lexical-proj-v2",
        )
        indexer.run_ingestion(chunk_size=320, chunk_overlap=80)

        rep_b = run_verification(
            corpus_dir=tmp_b,
            quarantine_dir=tmp_q,
            output_dir=tmp_path / "res",
            reference_corpus_dir=tmp_raw,
            archive_dir=tmp_a,
            batch_label="e3_batchB",
            embedding_version="lexical-proj-v2",
        )
        for rec in rep_b.records:
            if rec.action == "archived" and (tmp_raw / rec.filename).exists():
                indexer.archive_superseded(rec.filename)
        for passer in sorted(tmp_b.glob("*.md")):
            shutil.move(str(passer), str(tmp_raw / passer.name))
        indexer.run_ingestion(chunk_size=320, chunk_overlap=80)

        print("\n[E3 Step 1] Evaluating v1 golden.jsonl against drifted Batch-B index...")
        res_v1_drifted = run_evaluation_suite(
            golden_path="eval/golden.jsonl",
            vector_store_path=indexer.index_path,
            collection_name="fast-policies-drifted",
            embedding_version="lexical-proj-v2",
            label="e3_v1_on_drifted",
        )
        print_summary(res_v1_drifted)

        print("\n[E3 Step 2] Evaluating updated golden_v2.jsonl against drifted Batch-B index...")
        res_v2_drifted = run_evaluation_suite(
            golden_path="eval/golden_v2.jsonl",
            vector_store_path=indexer.index_path,
            collection_name="fast-policies-drifted",
            embedding_version="lexical-proj-v2",
            label="e3_v2_on_drifted",
        )
        print_summary(res_v2_drifted)


if __name__ == "__main__":
    main()
