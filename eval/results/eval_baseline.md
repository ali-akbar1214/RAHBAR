# Evaluation Summary: baseline
- **Timestamp**: 2026-11-23T20:00:10.970521+00:00
- **Git Commit**: `e523be9` | **Golden**: `golden.jsonl` (`9028e8666f75`)
- **Collection**: `fast-policies-v1` | **Embedding**: `lexical-proj-v1` | **Tuned Threshold (dev)**: `0.1648`

| Level | Metric | Value |
| :--- | :--- | :--- |
| Module | Paraphrase / Unrelated / Gap | 0.0974 / 0.0368 / +0.0607 |
| Module | Chunk Embed Mean / p95 (ms) | 0.454 / 0.685 |
| Module | Cross-Process Determinism Diff | 0.00000000 |
| Component | Doc Recall@3 / Recall@5 | 0.9500 / 0.9500 |
| Component | Page Recall@3 / Recall@5 | 0.8500 / 0.9000 |
| Component | MRR / nDCG@5 / Chunk nDCG@5 | 0.8417 / 0.8696 / 0.7738 |
| Component | Noisy Recall@5 (Delta) | 1.0000 (+0.0500) |
| End-to-End | Faithfulness (All / Test) | 0.8462 / 0.8125 |
| End-to-End | Answer Correctness (Keywords) | 0.8333 |
| End-to-End | Abstention / False Abstain | 1.0000 / 0.2000 |
| End-to-End | Citation Validity | 1.0000 |
| End-to-End | Latency p50 / p95 (ms) | 1033.0 / 3912.99 |
| End-to-End | LLM Fallback Count | 0 |
