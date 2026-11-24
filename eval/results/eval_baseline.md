# Evaluation Summary: baseline
- **Timestamp**: 2026-11-24T17:05:55.866487+00:00
- **Git Commit**: `b5f8d4b` | **Golden**: `golden.jsonl` (`9028e8666f75`)
- **Collection**: `fast-policies-v1` | **Embedding**: `lexical-proj-v1` | **Tuned Threshold (dev)**: `0.1648`

| Level | Metric | Value |
| :--- | :--- | :--- |
| Module | Paraphrase / Unrelated / Gap | 0.0974 / 0.0368 / +0.0607 |
| Module | Chunk Embed Mean / p95 (ms) | 0.475 / 0.615 |
| Module | Cross-Process Determinism Diff | 0.00000000 |
| Component | Doc Recall@3 / Recall@5 | 0.9500 / 0.9500 |
| Component | Page Recall@3 / Recall@5 | 0.8500 / 0.9000 |
| Component | MRR / nDCG@5 / Chunk nDCG@5 | 0.8417 / 0.8696 / 0.7738 |
| Component | Noise (+3 Docs) Recall@5 | 0.9500 (+0.0000) |
| Component | Noise (15% Typos) Recall@5 | 1.0000 (+0.0500) |
| Component | Combined Noise Recall@5 | 1.0000 (+0.0500) |
| End-to-End | Faithfulness (All / Test) | 0.8462 / 0.8125 |
| End-to-End | Answer Correctness (Keywords) | 0.8333 |
| End-to-End | Abstention / False Abstain | 1.0000 / 0.2000 |
| End-to-End | Citation Validity | 1.0000 |
| End-to-End | Latency p50 / p95 (ms) | 1222.49 / 3993.84 |
| End-to-End | LLM Fallback Count | 0 |
