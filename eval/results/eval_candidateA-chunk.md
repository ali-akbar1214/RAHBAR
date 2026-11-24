# Evaluation Summary: candidateA-chunk
- **Timestamp**: 2026-11-24T17:08:43.357435+00:00
- **Git Commit**: `b5f8d4b` | **Golden**: `golden.jsonl` (`9028e8666f75`)
- **Collection**: `cand-candidateA-chunk` | **Embedding**: `lexical-proj-v1` | **Tuned Threshold (dev)**: `0.1426`

| Level | Metric | Value |
| :--- | :--- | :--- |
| Module | Paraphrase / Unrelated / Gap | 0.0974 / 0.0368 / +0.0607 |
| Module | Chunk Embed Mean / p95 (ms) | 0.457 / 0.594 |
| Module | Cross-Process Determinism Diff | 0.00000000 |
| Component | Doc Recall@3 / Recall@5 | 0.9500 / 1.0000 |
| Component | Page Recall@3 / Recall@5 | 0.9000 / 0.9000 |
| Component | MRR / nDCG@5 / Chunk nDCG@5 | 0.8875 / 0.9162 / 0.1993 |
| Component | Noise (+3 Docs) Recall@5 | 1.0000 (+0.0000) |
| Component | Noise (15% Typos) Recall@5 | 1.0000 (+0.0000) |
| Component | Combined Noise Recall@5 | 1.0000 (+0.0000) |
| End-to-End | Faithfulness (All / Test) | 0.9231 / 0.8750 |
| End-to-End | Answer Correctness (Keywords) | 0.8910 |
| End-to-End | Abstention / False Abstain | 1.0000 / 0.1000 |
| End-to-End | Citation Validity | 1.0000 |
| End-to-End | Latency p50 / p95 (ms) | 3613.01 / 4944.07 |
| End-to-End | LLM Fallback Count | 0 |
