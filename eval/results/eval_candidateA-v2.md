# Evaluation Summary: candidateA-v2
- **Timestamp**: 2026-11-23T20:17:44.619562+00:00
- **Git Commit**: `e523be9` | **Golden**: `golden.jsonl` (`9028e8666f75`)
- **Collection**: `cand-candidateA-v2` | **Embedding**: `lexical-proj-v2` | **Tuned Threshold (dev)**: `0.2518`

| Level | Metric | Value |
| :--- | :--- | :--- |
| Module | Paraphrase / Unrelated / Gap | 0.1618 / 0.0033 / +0.1585 |
| Module | Chunk Embed Mean / p95 (ms) | 2.25 / 3.398 |
| Module | Cross-Process Determinism Diff | 0.00000000 |
| Component | Doc Recall@3 / Recall@5 | 1.0000 / 1.0000 |
| Component | Page Recall@3 / Recall@5 | 0.9500 / 0.9500 |
| Component | MRR / nDCG@5 / Chunk nDCG@5 | 0.9417 / 0.9565 / 0.2304 |
| Component | Noisy Recall@5 (Delta) | 1.0000 (+0.0000) |
| End-to-End | Faithfulness (All / Test) | 0.9231 / 0.8750 |
| End-to-End | Answer Correctness (Keywords) | 0.9103 |
| End-to-End | Abstention / False Abstain | 1.0000 / 0.1000 |
| End-to-End | Citation Validity | 1.0000 |
| End-to-End | Latency p50 / p95 (ms) | 1262.29 / 6077.53 |
| End-to-End | LLM Fallback Count | 0 |
