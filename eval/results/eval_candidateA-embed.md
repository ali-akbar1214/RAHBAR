# Evaluation Summary: candidateA-embed
- **Timestamp**: 2026-11-24T17:11:00.241329+00:00
- **Git Commit**: `b5f8d4b` | **Golden**: `golden.jsonl` (`9028e8666f75`)
- **Collection**: `cand-candidateA-embed` | **Embedding**: `lexical-proj-v2` | **Tuned Threshold (dev)**: `0.2464`

| Level | Metric | Value |
| :--- | :--- | :--- |
| Module | Paraphrase / Unrelated / Gap | 0.1618 / 0.0033 / +0.1585 |
| Module | Chunk Embed Mean / p95 (ms) | 0.807 / 0.93 |
| Module | Cross-Process Determinism Diff | 0.00000000 |
| Component | Doc Recall@3 / Recall@5 | 1.0000 / 1.0000 |
| Component | Page Recall@3 / Recall@5 | 0.9000 / 0.9500 |
| Component | MRR / nDCG@5 / Chunk nDCG@5 | 0.9417 / 0.9565 / 0.9327 |
| Component | Noise (+3 Docs) Recall@5 | 1.0000 (+0.0000) |
| Component | Noise (15% Typos) Recall@5 | 1.0000 (+0.0000) |
| Component | Combined Noise Recall@5 | 1.0000 (+0.0000) |
| End-to-End | Faithfulness (All / Test) | 0.8846 / 0.8750 |
| End-to-End | Answer Correctness (Keywords) | 0.8718 |
| End-to-End | Abstention / False Abstain | 1.0000 / 0.1500 |
| End-to-End | Citation Validity | 1.0000 |
| End-to-End | Latency p50 / p95 (ms) | 1428.74 / 4123.88 |
| End-to-End | LLM Fallback Count | 0 |
