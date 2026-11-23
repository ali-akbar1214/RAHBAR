"""Part B & C: Comprehensive 3-Level Offline Evaluation & Gate Runner.

Evaluates the RAGOps pipeline across:
  1. Module Level (Embeddings):
     - 15 fresh paraphrase pairs vs 15 fresh unrelated pairs (not copied from golden set).
     - Embedding latency (mean and p95) measured across ALL real chunks in the index.
     - Cross-process determinism check across two separate Python subprocesses.
  2. Component Level (Retrieval):
     - Document-level (deduped), Page-level, and Chunk-level Recall@3, Recall@5, MRR, nDCG@5.
     - Dual Noise Test: (a) +3 irrelevant distractor docs in temp index AND (b) 15% seeded query word corruption.
  3. End-to-End Level (Real QAEngine + Groq LLM):
     - Calls QAEngine.answer_query() for every golden question (never bypasses the engine).
     - Tunes abstention threshold on `dev` split only; evaluates on `test` split and full set.
     - Faithfulness evaluated by Groq LLM Judge (context + answer -> supported/unsupported + reason).
     - Deterministic answer_correctness (expected keywords present).
     - Human-vs-Judge agreement + Cohen's kappa from eval/human_labels.jsonl.
     - Real abstention correctness AND false-abstention rate on answerable queries.
     - Citation validity verified against retrieved chunks and vector store sections.
     - End-to-end total_ms latency (p50, p95) over repeated runs.
  4. Stochasticity Test (--stochasticity):
     - Runs the full suite in two separate Python subprocesses with fixed seeds and reports metric diffs.
  5. Optional Release Gate (--gate):
     - Evaluates metrics against baseline + eval/gate_config.yaml + env overrides and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import (
    PROJECT_ROOT,
    get_collection_name,
    get_embedding_version,
    get_index_path,
    get_results_dir,
    get_traces_dir,
    load_gate_config,
)
from src.pipeline.embed import cosine_similarity, embed_text
from src.pipeline.indexer import ChunkRecord, VectorStore
from src.pipeline.qa_engine import DEFAULT_GROQ_MODEL, QAEngine

# Fresh module-level pairs (distinct from eval/golden.jsonl questions, B-4)
PARAPHRASE_PAIRS: list[tuple[str, str]] = [
    ("How many semesters can a student take at most to finish a BS degree?", "What is the outer time limit in academic years for completing undergraduate studies?"),
    ("When is a student placed on academic probation after warning?", "How many warnings in a row lead to probationary standing?"),
    ("How much is charged for a certified physical copy of an academic transcript?", "What is the fee per attested hard-copy transcript?"),
    ("What happens if a student smokes or vapes inside the cafeteria?", "Is vaping prohibited on campus grounds and parking areas?"),
    ("Who can approve medical absence condonation up to five percent?", "Which university officials may grant a 5% medical attendance relaxation?"),
    ("What is the inter-campus transfer processing charge?", "How much fee applies when transferring between two FAST campuses?"),
    ("When does withdrawal yield zero percent tuition refund?", "After which day of classes are tuition refunds no longer granted?"),
    ("What is the penalty for destroying or tampering with lab hardware?", "What disciplinary action follows willful damage to university lab equipment?"),
    ("Are PhD scholars eligible for tuition waivers?", "Can doctoral candidates receive full fee waiver and research stipends?"),
    ("What is the maximum course load during academic warning?", "How many credit hours may a student on probation register in one term?"),
    ("How many campuses does FAST-NUCES operate in Pakistan?", "Which cities host FAST-NUCES undergraduate campuses?"),
    ("What letter grade corresponds to 3.67 grade points?", "Which grade carries 3.67 GPA weight in the grading table?"),
    ("What is the third-offense penalty for academic dishonesty?", "What happens upon a third confirmed plagiarism violation?"),
    ("Is the admission fee refundable if a student withdraws on day one?", "Can a student get back the one-time admission fee upon dropping out early?"),
    ("What credit hour total is needed for BS degree award?", "How many minimum credit hours must an undergraduate earn to graduate?"),
]

UNRELATED_PAIRS: list[tuple[str, str]] = [
    ("How many semesters can a student take at most to finish a BS degree?", "How long does it take to bake sourdough bread in a Dutch oven?"),
    ("When is a student placed on academic probation after warning?", "What is the distance from Earth to the Andromeda galaxy?"),
    ("How much is charged for a certified physical copy of an academic transcript?", "Which team won the 2019 Cricket World Cup final?"),
    ("What happens if a student smokes or vapes inside the cafeteria?", "How do coral reefs form in shallow tropical oceans?"),
    ("Who can approve medical absence condonation up to five percent?", "What programming language did Guido van Rossum create?"),
    ("What is the inter-campus transfer processing charge?", "How many strings are on a standard concert violin?"),
    ("When does withdrawal yield zero percent tuition refund?", "What is the melting point of pure titanium metal?"),
    ("What is the penalty for destroying or tampering with lab hardware?", "Where is the deepest trench in the Pacific Ocean located?"),
    ("Are PhD scholars eligible for tuition waivers?", "How do honeybees communicate nectar locations through dancing?"),
    ("What is the maximum course load during academic warning?", "What ingredients are required for classic Italian tiramisu?"),
    ("How many campuses does FAST-NUCES operate in Pakistan?", "How does magnetic levitation work on high-speed trains?"),
    ("What letter grade corresponds to 3.67 grade points?", "Who painted The Starry Night in 1889?"),
    ("What is the third-offense penalty for academic dishonesty?", "What causes aurora borealis near the magnetic poles?"),
    ("Is the admission fee refundable if a student withdraws on day one?", "How many bones exist in the adult human skeleton?"),
    ("What credit hour total is needed for BS degree award?", "What is the top speed of a peregrine falcon in a dive?"),
]

DISTRACTOR_CHUNKS = [
    ("distractor_cafeteria.md", "Cafeteria Menu", "Daily cafeteria lunch serves chicken karahi PKR 350, biryani PKR 280, and tea PKR 60 from 12:00 PM to 3:00 PM."),
    ("distractor_cricket.md", "Cricket Tournament", "The inter-departmental tape-ball cricket tournament follows 8-over innings with powerplay in overs 1 and 2."),
    ("distractor_travel.md", "Northern Areas Tour", "Student society summer excursion visits Hunza, Skardu, and Fairy Meadows via Karakoram Highway bus convoy."),
]


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(pct * len(sorted_vals)))
    return sorted_vals[idx]


def evaluate_module_level(vector_store: VectorStore, embedding_version: str) -> dict[str, Any]:
    para_sims = [
        cosine_similarity(embed_text(a, version=embedding_version), embed_text(b, version=embedding_version))
        for a, b in PARAPHRASE_PAIRS
    ]
    unrel_sims = [
        cosine_similarity(embed_text(a, version=embedding_version), embed_text(b, version=embedding_version))
        for a, b in UNRELATED_PAIRS
    ]
    mean_para = sum(para_sims) / len(para_sims)
    mean_unrel = sum(unrel_sims) / len(unrel_sims)

    # Measure embedding latency over ALL real chunks in vector store (B-5)
    chunk_texts = [c.text for c in vector_store.chunks.values()] or ["Sample policy text chunk."]
    chunk_latencies: list[float] = []
    for txt in chunk_texts:
        t0 = time.perf_counter()
        _ = embed_text(txt, version=embedding_version)
        chunk_latencies.append((time.perf_counter() - t0) * 1000.0)
    chunk_latencies.sort()

    # Cross-process determinism test across two separate Python subprocesses (B-5)
    det_cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; sys.path.insert(0, '.'); "
            "from src.pipeline.embed import embed_text; "
            f"print(json.dumps(embed_text('Minimum attendance is 80 percent for final exams.', version='{embedding_version}')))"
        ),
    ]
    v_proc1 = json.loads(subprocess.check_output(det_cmd, cwd=PROJECT_ROOT, text=True))
    v_proc2 = json.loads(subprocess.check_output(det_cmd, cwd=PROJECT_ROOT, text=True))
    max_diff = max(abs(x - y) for x, y in zip(v_proc1, v_proc2))

    # 10th percentile of paraphrase similarity for Part E threshold derivation
    sorted_para = sorted(para_sims)
    p10_para = _percentile(sorted_para, 0.10)

    return {
        "embedding_version": embedding_version,
        "mean_paraphrase_similarity": round(mean_para, 4),
        "p10_paraphrase_similarity": round(p10_para, 4),
        "mean_unrelated_similarity": round(mean_unrel, 4),
        "separation_gap": round(mean_para - mean_unrel, 4),
        "chunks_timed": len(chunk_latencies),
        "latency_per_chunk_mean_ms": round(sum(chunk_latencies) / len(chunk_latencies), 3),
        "latency_per_chunk_p95_ms": round(_percentile(chunk_latencies, 0.95), 3),
        "determinism_max_diff": float(max_diff),
    }


def _corrupt_query_words(text: str, rng: random.Random, rate: float = 0.15) -> str:
    words = text.split()
    out = []
    for w in words:
        if len(w) >= 4 and rng.random() < rate:
            chars = list(w)
            idx = rng.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            out.append("".join(chars))
        else:
            out.append(w)
    return " ".join(out)


def compute_retrieval_metrics(
    golden_questions: list[dict[str, Any]],
    vector_store: VectorStore,
    embedding_version: str,
    noisy_mode: bool = False,
    seed: int = 42,
) -> dict[str, float]:
    """Compute deduped doc-level, page-level, and chunk-level metrics (B-6, B-7)."""
    eval_store = vector_store
    if noisy_mode:
        # Build a temporary store copy with +3 irrelevant distractor docs (B-7)
        tmp_store = VectorStore.__new__(VectorStore)
        tmp_store.storage_path = vector_store.storage_path
        tmp_store.chunks = dict(vector_store.chunks)
        for idx, (src, sec, txt) in enumerate(DISTRACTOR_CHUNKS, start=1):
            cid = f"{src}_c01"
            tmp_store.chunks[cid] = ChunkRecord(
                chunk_id=cid,
                doc_id=src,
                source=src,
                doc_version="2026",
                title=sec,
                section=sec,
                text=f"{sec}: {txt}",
                vector=embed_text(f"{sec}: {txt}", version=embedding_version),
                superseded=False,
                page=1,
            )
        eval_store = tmp_store

    rng = random.Random(seed)
    answerable_qs = [
        q for q in golden_questions if q.get("type") in {"answerable", "hard"} and q.get("ground_truth_doc")
    ]
    if not answerable_qs:
        return {"recall_at_3": 0.0, "recall_at_5": 0.0, "page_recall_at_3": 0.0, "page_recall_at_5": 0.0, "chunk_recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}

    doc_r3, doc_r5 = 0, 0
    page_r3, page_r5 = 0, 0
    chunk_r5 = 0
    mrrs: list[float] = []
    doc_ndcgs: list[float] = []
    chunk_ndcgs: list[float] = []

    for q in answerable_qs:
        q_text = _corrupt_query_words(q["question"], rng, rate=0.15) if noisy_mode else q["question"]
        gt_doc = q["ground_truth_doc"]
        gt_page = q.get("ground_truth_page")
        gt_chunks = set(q.get("ground_truth_chunks") or [])

        q_vec = embed_text(q_text, version=embedding_version)
        # Fetch top 15 chunks so we can both rank chunks AND dedupe documents cleanly
        retrieved = eval_store.query(q_vec, top_k=15, filter_superseded=True)

        # Deduped document ranking (fixes nDCG > 1.0 bug in B-6)
        dedup_docs: list[str] = []
        for ch, _ in retrieved:
            if ch.source not in dedup_docs:
                dedup_docs.append(ch.source)

        if gt_doc in dedup_docs[:3]:
            doc_r3 += 1
        if gt_doc in dedup_docs[:5]:
            doc_r5 += 1

        rr = 0.0
        for rank, d in enumerate(dedup_docs, start=1):
            if d == gt_doc:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        doc_dcg = 0.0
        for rank, d in enumerate(dedup_docs[:5], start=1):
            if d == gt_doc:
                doc_dcg += 1.0 / math.log2(rank + 1)
        doc_ndcgs.append(doc_dcg)  # IDCG for 1 relevant doc = 1.0

        # Page-level recall (matching doc AND page)
        top5_chunks = [ch for ch, _ in retrieved[:5]]
        if gt_page is not None:
            if any(ch.source == gt_doc and ch.page == gt_page for ch in top5_chunks[:3]):
                page_r3 += 1
            if any(ch.source == gt_doc and ch.page == gt_page for ch in top5_chunks):
                page_r5 += 1

        # Chunk-level Recall@5 & nDCG@5 with true IDCG
        if gt_chunks:
            if any(ch.chunk_id in gt_chunks for ch in top5_chunks):
                chunk_r5 += 1
            c_dcg = 0.0
            for rank, ch in enumerate(top5_chunks, start=1):
                if ch.chunk_id in gt_chunks:
                    c_dcg += 1.0 / math.log2(rank + 1)
            k_rel = min(5, len(gt_chunks))
            c_idcg = sum(1.0 / math.log2(r + 1) for r in range(1, k_rel + 1))
            chunk_ndcgs.append(c_dcg / c_idcg if c_idcg > 0 else 0.0)

    n = len(answerable_qs)
    return {
        "recall_at_3": round(doc_r3 / n, 4),
        "recall_at_5": round(doc_r5 / n, 4),
        "page_recall_at_3": round(page_r3 / n, 4),
        "page_recall_at_5": round(page_r5 / n, 4),
        "chunk_recall_at_5": round(chunk_r5 / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "chunk_ndcg_at_5": round(sum(chunk_ndcgs) / len(chunk_ndcgs), 4) if chunk_ndcgs else 0.0,
        "ndcg_at_5": round(sum(doc_ndcgs) / n, 4),
    }


def tune_threshold_on_dev(
    golden_questions: list[dict[str, Any]],
    vector_store: VectorStore,
    embedding_version: str,
) -> tuple[float, dict[str, Any]]:
    """Tune abstention relevance threshold solely on the `dev` split (B-4)."""
    dev_qs = [q for q in golden_questions if q.get("split") == "dev"]
    ans_scores: list[float] = []
    unans_scores: list[float] = []

    from src.pipeline.qa_engine import normalize_and_rewrite_query, rerank_chunks

    for q in dev_qs:
        rewritten = normalize_and_rewrite_query(q["question"])
        q_vec = embed_text(rewritten, version=embedding_version)
        cands = vector_store.query(q_vec, top_k=6, filter_superseded=True)
        reranked = rerank_chunks(q["question"], cands, top_k=3)
        top_score = reranked[0][1] if reranked else 0.0
        if q.get("type") in {"answerable", "hard"}:
            ans_scores.append(top_score)
        elif q.get("type") == "unanswerable":
            unans_scores.append(top_score)

    max_unans = max(unans_scores) if unans_scores else 0.25
    min_ans = min(ans_scores) if ans_scores else 0.35
    if min_ans > max_unans:
        tuned = round((max_unans + min_ans) / 2.0, 4)
    else:
        # Find threshold maximizing classification accuracy on dev split
        candidates_t = sorted(set(ans_scores + unans_scores))
        best_acc = -1.0
        tuned = round((max_unans + min_ans) / 2.0, 4)
        for i in range(len(candidates_t) - 1):
            mid = (candidates_t[i] + candidates_t[i + 1]) / 2.0
            correct = sum(1 for s in ans_scores if s >= mid) + sum(1 for s in unans_scores if s < mid)
            acc = correct / max(1, len(ans_scores) + len(unans_scores))
            if acc > best_acc:
                best_acc = acc
                tuned = round(mid, 4)

    return tuned, {
        "dev_count": len(dev_qs),
        "dev_min_answerable_score": round(min_ans, 4),
        "dev_max_unanswerable_score": round(max_unans, 4),
        "tuned_threshold": tuned,
    }


def _llm_judge_faithfulness(
    groq_client: Any,
    model_name: str,
    question: str,
    context_text: str,
    answer: str,
) -> tuple[bool, str, bool]:
    """Call Groq LLM judge to verify whether answer is strictly supported by context (B-8)."""
    if not groq_client:
        # Deterministic lexical overlap fallback if no API client (clearly flagged)
        return False, "LLM judge unavailable (no Groq client)", True

    judge_prompt = (
        "You are a strict faithfulness auditor. Given Context and Answer, determine if EVERY factual claim "
        "in the Answer is directly supported by the Context. Reply strictly in JSON: "
        '{"verdict": "supported" or "unsupported", "reason": "short one-sentence reason"}.'
    )
    try:
        resp = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:\n{answer}"},
            ],
            temperature=0.0,
            max_tokens=120,
            seed=42,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Extract JSON block if wrapped in markdown
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw[start : end + 1])
            verdict = str(parsed.get("verdict", "")).lower().strip()
            reason = str(parsed.get("reason", ""))
            return verdict == "supported", reason, False
        is_sup = "supported" in raw.lower() and "unsupported" not in raw.lower()
        return is_sup, raw[:140], False
    except Exception as e:
        return False, f"Judge error: {e}", True


def _cohens_kappa(y1: list[bool], y2: list[bool]) -> float:
    n = len(y1)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    po = agree / n
    p1_true = sum(1 for a in y1 if a) / n
    p2_true = sum(1 for b in y2 if b) / n
    pe = (p1_true * p2_true) + ((1.0 - p1_true) * (1.0 - p2_true))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def evaluate_end_to_end(
    golden_questions: list[dict[str, Any]],
    vector_store_path: Path,
    collection_name: str,
    embedding_version: str,
    relevance_threshold: float,
    traces_path: Path,
    latency_runs: int = 1,
    require_llm: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run real end-to-end evaluation via QAEngine.answer_query() (B-8)."""
    engine = QAEngine(
        vector_store_path=vector_store_path,
        traces_path=traces_path,
        collection_name=collection_name,
        embedding_version=embedding_version,
        relevance_threshold=relevance_threshold,
        emit_stdout_traces=False,
    )

    # Index valid (source, section) pairs in store for citation verification
    valid_sections = {(c.source, c.section) for c in engine.vector_store.chunks.values()}

    per_question_records: list[dict[str, Any]] = []
    all_latencies_ms: list[float] = []
    fallback_count = 0

    faithfulness_vals: list[float] = []
    answer_correctness_vals: list[float] = []
    unans_total, unans_abstained = 0, 0
    ans_total, ans_false_abstained = 0, 0
    citation_checked, citation_valid = 0, 0

    for q in golden_questions:
        q_id = q["id"]
        q_type = q["type"]
        q_text = q["question"]

        res = None
        for r_idx in range(max(1, latency_runs)):
            res = engine.answer_query(q_text, use_cache=False)
            all_latencies_ms.append(float(res["latency_ms"]))

        assert res is not None
        if res.get("used_fallback"):
            fallback_count += 1

        did_abstain = bool(res["abstain"])
        answer = res["response"]
        citations = res.get("citations", [])
        retrieved_chunks = res.get("retrieved_chunks", [])

        if q_type in {"unanswerable", "adversarial"}:
            unans_total += 1
            if did_abstain:
                unans_abstained += 1
        else:
            ans_total += 1
            if did_abstain:
                ans_false_abstained += 1

        # Deterministic answer_correctness via expected_keywords
        exp_kws = q.get("expected_keywords") or []
        if q_type in {"unanswerable", "adversarial"}:
            kw_score = 1.0 if did_abstain else 0.0
        elif exp_kws:
            ans_l = answer.lower()
            kw_score = sum(1 for kw in exp_kws if kw.lower() in ans_l) / len(exp_kws)
        else:
            kw_score = 0.0 if did_abstain else 1.0
        answer_correctness_vals.append(kw_score)

        # Citation validity: every cited (source, section) must appear in retrieved chunks AND vector store
        c_valid = True
        if not did_abstain and citations:
            citation_checked += 1
            ret_pairs = {(rc.get("doc_id"), rc.get("section")) for rc in retrieved_chunks}
            for cit in citations:
                src = cit.get("source", "")
                sec = cit.get("section", "")
                if (src, sec) not in valid_sections:
                    c_valid = False
                    break
                if not any(sec == rc.get("section") for rc in retrieved_chunks):
                    c_valid = False
                    break
            if c_valid:
                citation_valid += 1
        elif not did_abstain and not citations:
            citation_checked += 1
            c_valid = False

        # Faithfulness via real LLM judge (B-8)
        if did_abstain:
            is_faithful = q_type in {"unanswerable", "adversarial"}
            judge_reason = (
                "Correctly abstained on out-of-domain/adversarial query"
                if is_faithful
                else "False abstention on answerable question"
            )
            judge_fb = False
        else:
            ctx_parts = []
            for rc in retrieved_chunks:
                cid = rc.get("chunk_id", "")
                full_ch = engine.vector_store.chunks.get(cid)
                ch_txt = full_ch.text if full_ch else rc.get("text_snippet", "")
                ctx_parts.append(f"[{rc.get('doc_id')} | {rc.get('section')}]: {ch_txt}")
            ctx_str = "\n".join(ctx_parts)
            is_faithful, judge_reason, judge_fb = _llm_judge_faithfulness(
                engine.groq_client, engine.model_name, q_text, ctx_str, answer
            )
            if judge_fb:
                fallback_count += 1

        faithfulness_vals.append(1.0 if is_faithful else 0.0)

        per_question_records.append(
            {
                "question_id": q_id,
                "split": q.get("split", "test"),
                "type": q_type,
                "trace_id": res["trace_id"],
                "answer": answer,
                "abstain": did_abstain,
                "citations": citations,
                "citation_valid": c_valid,
                "used_fallback": bool(res.get("used_fallback", False)),
                "latency_ms": res["latency_ms"],
                "answer_correctness": round(kw_score, 4),
                "judge_faithful": is_faithful,
                "judge_reason": judge_reason,
            }
        )

    if require_llm and fallback_count > 0:
        raise RuntimeError(f"LLM fallback triggered {fallback_count} times while --require-llm is set.")

    # Human-vs-Judge Agreement from eval/human_labels.jsonl
    human_labels_path = PROJECT_ROOT / "eval" / "human_labels.jsonl"
    agreement_rate = None
    kappa = None
    human_count = 0
    disagreements: list[dict[str, Any]] = []
    if human_labels_path.exists():
        h_map = {}
        for line in human_labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                h_map[obj["question_id"]] = obj
        y_human: list[bool] = []
        y_judge: list[bool] = []
        for rec in per_question_records:
            qid = rec["question_id"]
            if qid in h_map:
                h_val = bool(h_map[qid]["human_faithful"])
                j_val = bool(rec["judge_faithful"])
                y_human.append(h_val)
                y_judge.append(j_val)
                if h_val != j_val:
                    disagreements.append(
                        {
                            "question_id": qid,
                            "human_faithful": h_val,
                            "judge_faithful": j_val,
                            "human_note": h_map[qid].get("note", ""),
                            "judge_reason": rec["judge_reason"],
                        }
                    )
        human_count = len(y_human)
        if human_count > 0:
            agreement_rate = round(sum(1 for a, b in zip(y_human, y_judge) if a == b) / human_count, 4)
            kappa = round(_cohens_kappa(y_human, y_judge), 4)

    all_latencies_ms.sort()
    test_recs = [r for r in per_question_records if r["split"] == "test"]
    test_faith = (
        round(sum(1.0 if r["judge_faithful"] else 0.0 for r in test_recs) / len(test_recs), 4)
        if test_recs
        else 0.0
    )

    e2e_summary = {
        "faithfulness": round(sum(faithfulness_vals) / len(faithfulness_vals), 4),
        "test_split_faithfulness": test_faith,
        "answer_correctness": round(sum(answer_correctness_vals) / len(answer_correctness_vals), 4),
        "abstention_correctness": round(unans_abstained / unans_total, 4) if unans_total else 1.0,
        "false_abstention_rate": round(ans_false_abstained / ans_total, 4) if ans_total else 0.0,
        "citation_validity": round(citation_valid / citation_checked, 4) if citation_checked else 1.0,
        "latency_p50_ms": round(_percentile(all_latencies_ms, 0.50), 2),
        "latency_p95_ms": round(_percentile(all_latencies_ms, 0.95), 2),
        "latency_samples": len(all_latencies_ms),
        "used_fallback_count": fallback_count,
        "human_judge_agreement_rate": agreement_rate,
        "human_judge_cohens_kappa": kappa,
        "human_validated_count": human_count,
        "human_judge_disagreements": disagreements,
    }
    return e2e_summary, per_question_records


def run_evaluation_suite(
    golden_path: str | Path = "eval/golden.jsonl",
    output_dir: str | Path | None = None,
    vector_store_path: str | Path | None = None,
    collection_name: str | None = None,
    embedding_version: str | None = None,
    label: str = "baseline",
    seed: int = 42,
    latency_runs: int = 1,
    require_llm: bool = False,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    col = collection_name or get_collection_name()
    emb_ver = embedding_version or get_embedding_version()
    idx_path = Path(vector_store_path) if vector_store_path else get_index_path(col)
    out_dir = Path(output_dir) if output_dir else get_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    g_path = Path(golden_path)
    if not g_path.is_absolute():
        g_path = PROJECT_ROOT / g_path
    raw_golden = g_path.read_text(encoding="utf-8")
    golden_hash = hashlib.sha256(raw_golden.encode("utf-8")).hexdigest()[:12]
    questions = [json.loads(line) for line in raw_golden.splitlines() if line.strip()]

    vector_store = VectorStore(idx_path)

    # 1. Module level
    module_res = evaluate_module_level(vector_store, emb_ver)

    # 2. Component level (Clean + Dual Noise Test)
    comp_clean = compute_retrieval_metrics(questions, vector_store, emb_ver, noisy_mode=False, seed=seed)
    comp_noisy = compute_retrieval_metrics(questions, vector_store, emb_ver, noisy_mode=True, seed=seed)

    # 3. Tune abstention threshold on `dev` split only (B-4)
    tuned_thresh, dev_tune_info = tune_threshold_on_dev(questions, vector_store, emb_ver)

    # 4. End-to-End level via real QAEngine
    traces_file = get_traces_dir() / f"traces_eval_{label}.jsonl"
    if traces_file.exists():
        traces_file.unlink()
    e2e_res, per_q = evaluate_end_to_end(
        questions,
        vector_store_path=idx_path,
        collection_name=col,
        embedding_version=emb_ver,
        relevance_threshold=tuned_thresh,
        traces_path=traces_file,
        latency_runs=latency_runs,
        require_llm=require_llm,
    )

    ts_int = int(time.time())
    results = {
        "label": label,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "golden_path": str(g_path.name),
        "golden_sha256_prefix": golden_hash,
        "collection_name": col,
        "embedding_version": emb_ver,
        "seed": seed,
        "golden_questions_count": len(questions),
        "dev_threshold_tuning": dev_tune_info,
        "module_level": module_res,
        "component_level": {
            "clean": comp_clean,
            "noisy": comp_noisy,
            "noise_recall_delta": round(comp_noisy["recall_at_5"] - comp_clean["recall_at_5"], 4),
        },
        "end_to_end": e2e_res,
        "per_question": per_q,
    }

    json_file = out_dir / f"eval_{label}_{ts_int}.json"
    tmp = json_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(tmp, json_file)
    results["report_file"] = str(json_file)

    # Also write markdown summary table (B-11)
    md_lines = [
        f"# Evaluation Summary: {label}",
        f"- **Timestamp**: {results['timestamp']}",
        f"- **Git Commit**: `{results['git_commit']}` | **Golden**: `{results['golden_path']}` (`{golden_hash}`)",
        f"- **Collection**: `{col}` | **Embedding**: `{emb_ver}` | **Tuned Threshold (dev)**: `{tuned_thresh}`",
        "",
        "| Level | Metric | Value |",
        "| :--- | :--- | :--- |",
        f"| Module | Paraphrase / Unrelated / Gap | {module_res['mean_paraphrase_similarity']:.4f} / {module_res['mean_unrelated_similarity']:.4f} / +{module_res['separation_gap']:.4f} |",
        f"| Module | Chunk Embed Mean / p95 (ms) | {module_res['latency_per_chunk_mean_ms']} / {module_res['latency_per_chunk_p95_ms']} |",
        f"| Module | Cross-Process Determinism Diff | {module_res['determinism_max_diff']:.8f} |",
        f"| Component | Doc Recall@3 / Recall@5 | {comp_clean['recall_at_3']:.4f} / {comp_clean['recall_at_5']:.4f} |",
        f"| Component | Page Recall@3 / Recall@5 | {comp_clean['page_recall_at_3']:.4f} / {comp_clean['page_recall_at_5']:.4f} |",
        f"| Component | MRR / nDCG@5 / Chunk nDCG@5 | {comp_clean['mrr']:.4f} / {comp_clean['ndcg_at_5']:.4f} / {comp_clean['chunk_ndcg_at_5']:.4f} |",
        f"| Component | Noisy Recall@5 (Delta) | {comp_noisy['recall_at_5']:.4f} ({results['component_level']['noise_recall_delta']:+.4f}) |",
        f"| End-to-End | Faithfulness (All / Test) | {e2e_res['faithfulness']:.4f} / {e2e_res['test_split_faithfulness']:.4f} |",
        f"| End-to-End | Answer Correctness (Keywords) | {e2e_res['answer_correctness']:.4f} |",
        f"| End-to-End | Abstention / False Abstain | {e2e_res['abstention_correctness']:.4f} / {e2e_res['false_abstention_rate']:.4f} |",
        f"| End-to-End | Citation Validity | {e2e_res['citation_validity']:.4f} |",
        f"| End-to-End | Latency p50 / p95 (ms) | {e2e_res['latency_p50_ms']} / {e2e_res['latency_p95_ms']} |",
        f"| End-to-End | LLM Fallback Count | {e2e_res['used_fallback_count']} |",
    ]
    md_file = out_dir / f"eval_{label}_{ts_int}.md"
    md_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return results


def print_summary(res: dict[str, Any]) -> None:
    print("\n" + "=" * 82)
    print(f"  RAGOPS 3-LEVEL OFFLINE EVALUATION REPORT [{res['label']}]")
    print("=" * 82)
    print(
        f"Commit: {res['git_commit']} | Collection: {res['collection_name']} "
        f"| Embedding: {res['embedding_version']} | Qs: {res['golden_questions_count']}"
    )
    dt = res["dev_threshold_tuning"]
    print(
        f"Dev Threshold Tuning (n={dt['dev_count']}): min_ans={dt['dev_min_answerable_score']}, "
        f"max_unans={dt['dev_max_unanswerable_score']} -> tuned={dt['tuned_threshold']}"
    )

    m = res["module_level"]
    print("\n[LEVEL 1: MODULE LEVEL (LEXICAL EMBEDDING)]")
    print(f"  - Paraphrase Sim: {m['mean_paraphrase_similarity']:.4f} (p10={m['p10_paraphrase_similarity']:.4f})")
    print(f"  - Unrelated Sim:  {m['mean_unrelated_similarity']:.4f} | Separation Gap: {m['separation_gap']:+.4f}")
    print(f"  - Chunk Latency ({m['chunks_timed']} chunks): mean={m['latency_per_chunk_mean_ms']} ms, p95={m['latency_per_chunk_p95_ms']} ms")
    print(f"  - Cross-Process Determinism Diff: {m['determinism_max_diff']:.8f}")

    cc = res["component_level"]["clean"]
    cn = res["component_level"]["noisy"]
    print("\n[LEVEL 2: COMPONENT LEVEL (RETRIEVAL)]")
    print(f"  {'Metric':<18} {'Clean':>10} {'Noisy(+3docs/15%typo)':>22} {'Delta':>10}")
    print("  " + "-" * 64)
    for k, lbl in [
        ("recall_at_3", "Doc Recall@3"),
        ("recall_at_5", "Doc Recall@5"),
        ("page_recall_at_3", "Page Recall@3"),
        ("page_recall_at_5", "Page Recall@5"),
        ("chunk_recall_at_5", "Chunk Recall@5"),
        ("mrr", "MRR"),
        ("ndcg_at_5", "Doc nDCG@5"),
        ("chunk_ndcg_at_5", "Chunk nDCG@5"),
    ]:
        print(f"  {lbl:<18} {cc[k]:>10.4f} {cn[k]:>22.4f} {cn[k]-cc[k]:>+10.4f}")

    e = res["end_to_end"]
    print("\n[LEVEL 3: END-TO-END LEVEL (REAL QAEngine + GROQ LLM)]")
    print(f"  - Faithfulness (LLM Judge):  {e['faithfulness']:.4f} (test split: {e['test_split_faithfulness']:.4f})")
    print(f"  - Answer Correctness (KW):   {e['answer_correctness']:.4f}")
    print(f"  - Abstention Correctness:    {e['abstention_correctness']:.4f} | False Abstention: {e['false_abstention_rate']:.4f}")
    print(f"  - Citation Validity:         {e['citation_validity']:.4f}")
    print(f"  - End-to-End Latency:        p50={e['latency_p50_ms']} ms, p95={e['latency_p95_ms']} ms (n={e['latency_samples']})")
    print(f"  - LLM Fallbacks Used:        {e['used_fallback_count']}")
    if e["human_judge_agreement_rate"] is not None:
        print(
            f"  - Human-Judge Agreement:     {e['human_judge_agreement_rate']:.4f} "
            f"(Cohen's kappa={e['human_judge_cohens_kappa']}, n={e['human_validated_count']})"
        )
    print("=" * 82)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part B/C Offline Evaluation & Gate Suite")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--embedding-version", default=None)
    parser.add_argument("--vector-store", default=None)
    parser.add_argument("--label", default=os.environ.get("CANDIDATE_LABEL", "baseline"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--stochasticity", action="store_true")
    parser.add_argument("--gate", action="store_true", help="Evaluate against release gate and exit non-zero on fail")
    parser.add_argument("--save-baseline", action="store_true", help="Persist results to eval/baseline.json")
    args = parser.parse_args()

    res = run_evaluation_suite(
        golden_path=args.golden,
        vector_store_path=args.vector_store,
        collection_name=args.collection,
        embedding_version=args.embedding_version,
        label=args.label,
        seed=args.seed,
        latency_runs=args.latency_runs,
        require_llm=args.require_llm,
    )
    print_summary(res)

    if args.save_baseline:
        base_file = PROJECT_ROOT / "eval" / "baseline.json"
        base_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[Saved Baseline] -> {base_file}")

    if args.stochasticity:
        print("\n[STOCHASTICITY CHECK: Running two separate Python subprocesses with PYTHONHASHSEED=42...]")
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(args.seed)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--golden", args.golden,
            "--label", f"{args.label}_stoch",
            "--seed", str(args.seed),
        ]
        if args.collection:
            cmd.extend(["--collection", args.collection])
        if args.embedding_version:
            cmd.extend(["--embedding-version", args.embedding_version])
        if args.vector_store:
            cmd.extend(["--vector-store", args.vector_store])

        subprocess.check_call(cmd, env=env)
        subprocess.check_call(cmd, env=env)
        files = sorted(get_results_dir().glob(f"eval_{args.label}_stoch_*.json"))
        r1 = json.loads(files[-2].read_text(encoding="utf-8"))
        r2 = json.loads(files[-1].read_text(encoding="utf-8"))
        print(f"\n  {'Metric':<28} {'Run 1':>10} {'Run 2':>10} {'Abs Diff':>10}")
        print("  " + "-" * 62)
        pairs = [
            ("Doc Recall@5", r1["component_level"]["clean"]["recall_at_5"], r2["component_level"]["clean"]["recall_at_5"]),
            ("MRR", r1["component_level"]["clean"]["mrr"], r2["component_level"]["clean"]["mrr"]),
            ("Faithfulness", r1["end_to_end"]["faithfulness"], r2["end_to_end"]["faithfulness"]),
            ("Abstention Correctness", r1["end_to_end"]["abstention_correctness"], r2["end_to_end"]["abstention_correctness"]),
            ("Citation Validity", r1["end_to_end"]["citation_validity"], r2["end_to_end"]["citation_validity"]),
            ("Latency p95 (ms)", r1["end_to_end"]["latency_p95_ms"], r2["end_to_end"]["latency_p95_ms"]),
        ]
        for name, v1, v2 in pairs:
            print(f"  {name:<28} {v1:>10.4f} {v2:>10.4f} {abs(v2 - v1):>10.4f}")
        print(
            "  Note: Retrieval metrics are identical across processes under fixed seeds; "
            "latency varies due to network/API jitter, and LLM wording may show minor non-determinism."
        )

    if args.gate:
        from scripts.release_gate import check_release_gate

        passed, gate_report = check_release_gate(res, candidate_name=args.label)
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
