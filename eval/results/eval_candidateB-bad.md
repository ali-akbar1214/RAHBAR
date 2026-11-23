# Evaluation Summary: candidateB-bad
- **Timestamp**: 2026-11-23T20:18:20.271183+00:00
- **Git Commit**: `e523be9` | **Golden**: `golden.jsonl` (`9028e8666f75`)
- **Collection**: `cand-candidateB-bad` | **Embedding**: `lexical-proj-v1` | **Tuned Threshold (dev)**: `0.263`

| Level | Metric | Value |
| :--- | :--- | :--- |
| Module | Paraphrase / Unrelated / Gap | 0.0974 / 0.0368 / +0.0607 |
| Module | Chunk Embed Mean / p95 (ms) | 0.902 / 1.72 |
| Module | Cross-Process Determinism Diff | 0.00000000 |
| Component | Doc Recall@3 / Recall@5 | 0.9500 / 1.0000 |
| Component | Page Recall@3 / Recall@5 | 0.8500 / 0.8500 |
| Component | MRR / nDCG@5 / Chunk nDCG@5 | 0.8792 / 0.9096 / 0.0334 |
| Component | Noisy Recall@5 (Delta) | 1.0000 (+0.0000) |
| End-to-End | Faithfulness (All / Test) | 0.2308 / 0.1875 |
| End-to-End | Answer Correctness (Keywords) | 0.2628 |
| End-to-End | Abstention / False Abstain | 1.0000 / 1.0000 |
| End-to-End | Citation Validity | 1.0000 |
| End-to-End | Latency p50 / p95 (ms) | 544.9 / 2699.45 |
| End-to-End | LLM Fallback Count | 0 |
