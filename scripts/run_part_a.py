"""Part A End-to-End Verification & Incremental Ingestion Demo (A1 - A11).

Runs all mutation experiments (modify, delete, archive, rollback) inside an
isolated tempfile.TemporaryDirectory() so production corpus/raw and data/ are
never corrupted (A-9), while writing labelled verification reports to
eval/results/ and building the clean baseline fast-policies-v1 collection.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import get_index_dir, get_raw_dir, get_results_dir
from src.pipeline.embed import embed_text
from src.pipeline.indexer import IncrementalIndexer
from src.pipeline.verification import run_verification


def _print_counts_table(title: str, before: dict[str, int], after: dict[str, int]) -> None:
    print(f"\n  [{title}]")
    all_docs = sorted(set(before.keys()) | set(after.keys()))
    print(f"  {'Document ID':<32} | {'Before':>6} | {'After':>6} | {'Delta':>6}")
    print("  " + "-" * 58)
    for d in all_docs:
        b = before.get(d, 0)
        a = after.get(d, 0)
        delta = a - b
        d_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {d:<32} | {b:>6} | {a:>6} | {d_str:>6}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_raw = get_raw_dir()
    src_batch_b = project_root / "corpus" / "batch_b"
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Archive any old stale verification_report_* files into dev_history/ (A-11)
    dev_hist = results_dir / "dev_history"
    for old_rep in results_dir.glob("verification_report_*.json"):
        dev_hist.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_rep), str(dev_hist / old_rep.name))

    print("=" * 72)
    print("PART A: DATA MANAGEMENT, VERIFICATION & INCREMENTAL INGESTION")
    print("=" * 72)

    # 1. Build/sync clean baseline collection in data/fast-policies-v1.json
    base_indexer = IncrementalIndexer(
        corpus_dir=src_raw,
        archive_dir=project_root / "corpus" / "archive",
        data_dir=get_index_dir(),
        collection_name="fast-policies-v1",
    )
    base_stats = base_indexer.run_ingestion()
    print(f"\n[Baseline Sync] fast-policies-v1 active chunks: {base_stats['active_chunks_in_store']}")

    # 2. Run full Part A1 + A2 experiments in an isolated temporary workspace (A-9)
    with tempfile.TemporaryDirectory(prefix="ragops_part_a_") as tmp:
        tmp_path = Path(tmp)
        tmp_raw = tmp_path / "raw"
        tmp_batch_b = tmp_path / "batch_b"
        tmp_quarantine = tmp_path / "quarantine"
        tmp_archive = tmp_path / "archive"
        tmp_data = tmp_path / "data"

        shutil.copytree(src_raw, tmp_raw)
        shutil.copytree(src_batch_b, tmp_batch_b)

        # --- Step A1-A: Verify Batch A ---
        print("\n[Step 1] Verification on Batch A (5 baseline files)...")
        rep_a = run_verification(
            corpus_dir=tmp_raw,
            quarantine_dir=tmp_quarantine,
            output_dir=results_dir,
            archive_dir=tmp_archive,
            batch_label="batchA",
        )
        print(
            f"  Scanned={rep_a.total_scanned}, Accepted={rep_a.passed_count}, "
            f"Quarantined={rep_a.quarantined_count}, Archived={rep_a.archived_count}"
        )
        pcs_a = rep_a.per_check_summary
        print(
            f"  Checks: Q({pcs_a['quality']['passed']}/{pcs_a['quality']['failed']}), "
            f"C({pcs_a['completeness']['passed']}/{pcs_a['completeness']['failed']}), "
            f"R({pcs_a['recency']['passed']}/{pcs_a['recency']['failed']}), "
            f"U({pcs_a['uniqueness']['passed']}/{pcs_a['uniqueness']['failed']})"
        )

        # --- Step A2-1: Initial Ingestion of Batch A ---
        indexer = IncrementalIndexer(
            corpus_dir=tmp_raw,
            archive_dir=tmp_archive,
            data_dir=tmp_data,
            collection_name="fast-policies-v1",
        )
        st_a1 = indexer.run_ingestion()
        print(f"\n[Step 2] Batch A Initial Ingestion: added={st_a1['added']}, chunks_embedded={st_a1['chunks_embedded']}")

        # --- Step A1-B: Verify Batch B against Batch A ---
        print("\n[Step 3] Verification on Batch B (10 files with planted defects & newer 2027 rules)...")
        rep_b = run_verification(
            corpus_dir=tmp_batch_b,
            quarantine_dir=tmp_quarantine,
            output_dir=results_dir,
            reference_corpus_dir=tmp_raw,
            archive_dir=tmp_archive,
            batch_label="batchB",
        )
        print(
            f"  Scanned={rep_b.total_scanned}, Accepted={rep_b.passed_count}, "
            f"Quarantined={rep_b.quarantined_count}, Archived={rep_b.archived_count}"
        )
        pcs_b = rep_b.per_check_summary
        print(
            f"  Checks: Q({pcs_b['quality']['passed']}/{pcs_b['quality']['failed']}), "
            f"C({pcs_b['completeness']['passed']}/{pcs_b['completeness']['failed']}), "
            f"R({pcs_b['recency']['passed']}/{pcs_b['recency']['failed']}), "
            f"U({pcs_b['uniqueness']['passed']}/{pcs_b['uniqueness']['failed']})"
        )
        print("\n  Batch B Verification Verdicts & Max Similarity:")
        for r in rep_b.records:
            sim_info = f"max_sim={r.max_similarity:.4f} ({r.most_similar_doc})" if r.most_similar_doc else "-"
            reason = r.quarantine_reason or "All checks passed"
            print(f"    - {r.filename:<30} | {r.action.upper():<11} | {sim_info}")
            print(f"        -> {reason}")

        # --- Step A2-2: Ingest Batch B Passers into Lake & Archive Superseded (A-2) ---
        for rec in rep_b.records:
            if rec.action == "archived" and (tmp_raw / rec.filename).exists():
                indexer.archive_superseded(rec.filename)
        for passer in sorted(tmp_batch_b.glob("*.md")):
            shutil.move(str(passer), str(tmp_raw / passer.name))

        before_b = indexer.vector_store.chunk_counts_by_doc(include_superseded=True)
        st_b = indexer.run_ingestion()
        after_b = indexer.vector_store.chunk_counts_by_doc(include_superseded=True)
        print(
            f"\n[Step 4] Batch B Ingestion (A-2): added={st_b['added']}, "
            f"chunks_embedded={st_b['chunks_embedded']}, archived_preserved={st_b['archived_preserved']}"
        )
        _print_counts_table("Batch B Ingestion Chunk Counts", before_b, after_b)

        # --- Step A2-3: Re-run on Unchanged Corpus -> 0 embedded ---
        st_unchanged = indexer.run_ingestion()
        print(f"\n[Step 5] Unchanged Corpus Re-run: chunks_embedded={st_unchanged['chunks_embedded']} (expected 0)")
        assert st_unchanged["chunks_embedded"] == 0, "Expected 0 chunks embedded on unchanged corpus!"

        # --- Step A2-4: Modify one file -> only its chunks replaced ---
        fee_file = tmp_raw / "fee-structure-2026.md"
        before_mod = indexer.vector_store.chunk_counts_by_doc(include_superseded=True)
        fee_file.write_text(
            fee_file.read_text(encoding="utf-8")
            + "\n3.4 Emergency Medical Withdrawal: Full pro-rata refund upon Medical Board approval.\n",
            encoding="utf-8",
        )
        st_mod = indexer.run_ingestion()
        after_mod = indexer.vector_store.chunk_counts_by_doc(include_superseded=True)
        print(f"\n[Step 6] Modified fee-structure-2026.md: modified={st_mod['modified']}, chunks_embedded={st_mod['chunks_embedded']}")
        _print_counts_table("Single Document Modification", before_mod, after_mod)

        # --- Step A2-5: Delete one file -> 0 orphan chunks ---
        before_del = indexer.vector_store.chunk_counts_by_doc(include_superseded=True)
        (tmp_raw / "prospectus-2026.md").unlink()
        st_del = indexer.run_ingestion()
        after_del = indexer.vector_store.chunk_counts_by_doc(include_superseded=True)
        print(f"\n[Step 7] Deleted prospectus-2026.md: deleted={st_del['deleted']}, chunks_deleted={st_del['chunks_deleted']}")
        _print_counts_table("Single Document Deletion (0 Orphans)", before_del, after_del)
        assert indexer.vector_store.count_by_doc_id("prospectus-2026.md") == 0

        # --- Step A2-6: Archive -> Re-ingest (A-3 check) -> Rollback ---
        target_doc = "student-handbook-2026.md"
        q_vec = embed_text("plagiarism disciplinary penalty")
        active_before = indexer.vector_store.count_by_doc_id(target_doc, include_superseded=False)
        indexer.archive_superseded(target_doc)
        st_post_arch = indexer.run_ingestion()  # A-3 regression: must NOT delete archived chunks!
        total_after_arch = indexer.vector_store.count_by_doc_id(target_doc, include_superseded=True)
        active_after_arch = indexer.vector_store.count_by_doc_id(target_doc, include_superseded=False)
        hits_excluded = [c.doc_id for c, _ in indexer.vector_store.query(q_vec, top_k=5, filter_superseded=True)]
        hits_unfiltered = [c.doc_id for c, _ in indexer.vector_store.query(q_vec, top_k=5, filter_superseded=False)]

        print(
            f"\n[Step 8] Archive + Re-ingest (A-3 Regression): "
            f"active_before={active_before}, total_preserved={total_after_arch}, active_after={active_after_arch}"
        )
        print(f"  Retrieval with filter_superseded=True:  {hits_excluded}")
        print(f"  Retrieval with filter_superseded=False: {hits_unfiltered}")
        assert total_after_arch == active_before and active_after_arch == 0

        indexer.rollback_version(target_doc)
        indexer.run_ingestion()
        active_after_rb = indexer.vector_store.count_by_doc_id(target_doc, include_superseded=False)
        print(f"[Step 9] Rollback + Re-ingest: active_after_rollback={active_after_rb}")
        assert active_after_rb == active_before

    print("\nPart A completed cleanly with 0 production mutations.")


if __name__ == "__main__":
    main()
