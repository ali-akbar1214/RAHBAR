# RAHBAR — RAGOps Lifecycle Management Pipeline

[![Course: CS4032](https://img.shields.io/badge/Course-CS4032_Software_Construction_&_Development-blue.svg)](https://github.com/ali-akbar1214/RAHBAR)
[![Platform: Kubernetes](https://img.shields.io/badge/Platform-Kubernetes_Minikube-326ce5.svg)](k8s/)

**Institution**: National University of Computer & Emerging Sciences (FAST-NUCES), Islamabad  
**Department**: Department of Software Engineering  
**Course**: Software Construction and Development (CS4032) — Assignment 03  
**Repository**: `https://github.com/ali-akbar1214/RAHBAR`  
**Team Members**:
- **Syed Ali Akbar Naqvi** (24I-3079) — `24i3079@isb.nu.edu.pk`
- **Huzaifa Ali** (24I-3114) — `24i3114@isb.nu.edu.pk`

---

## 1. Architecture & Provenance

**RAHBAR** implements the **RAGOps Lifecycle Management System** (*Xu et al., 2026*) over FAST-NUCES regulatory documents:
- **Corpus Provenance**: Documents in `corpus/raw/` and `corpus/batch_b/` are condensed, structured Markdown excerpts of FAST-NUCES policy regulations (academic rules, fee structure, prospectus, student handbook, AI ethics, hostel/transport) annotated with `<!-- page: N -->` page markers, plus planted defect files (`doc_b01`–`doc_b08`) for verification testing.
- **Embeddings**: Deterministic versioned **lexical projection embeddings** (`lexical-proj-v1` dim=256 word+bigram; `lexical-proj-v2` dim=384 word+bigram+char-4gram) implemented in pure NumPy (`src/pipeline/embed.py`).
- **Part A (Data Management)**: 4-way pre-chunking verification (`Quality`, `Completeness`, `doc_family` `Recency`, whole-body `Uniqueness`) routing defects to quarantine and superseded editions to archive; hash-diff incremental updater (`0` chunks embedded on unchanged corpus); atomic writes; orphan elimination; rollback.
- **Part B (Offline 3-Level Testing)**: Module, Component (doc/page/chunk Recall@k, MRR, nDCG@5, dual noise test), and real End-to-End evaluation via `QAEngine.answer_query()` + Groq `qwen/qwen3.8-27b` LLM & LLM Judge + `eval/human_labels.jsonl`.
- **Part C (Change Experiment & Release Gate)**: Automated gate comparing isolated candidate collections against `eval/baseline.json` and `eval/gate_config.yaml`.
- **Part D (Observability & Traceability)**: Bare JSONL query traces (`event: "trace"`), unified input + retrieval injection guardrails (`src/pipeline/guardrails.py`), semantic cache, CLI trace inspector, and time-windowed telemetry.
- **Part E (Drift & Coverage)**: Part-B-derived coverage threshold (`0.1618`), pure-NumPy agglomerative clustering over all uncovered live queries (`82.5% -> 90.0%`), Batch B data drift evaluation, and versioned `eval/golden_v2.jsonl`.
- **Part F (Kubernetes Deployment)**: Numbered Minikube manifests with `ragops-data-pvc` sole-writer enforcement (`readOnly: true` on API/Eval), per-workload resource sizing, ConfigMap gate overrides, rolling updates, and `kubectl rollout undo`.

---

## 2. Repository Structure

```text
RAHBAR/
├── corpus/
│   ├── raw/                  # Active verified baseline corpus (5 policy documents)
│   └── batch_b/              # Planted defects + valid additions (10 documents)
├── data/
│   └── fast-policies-v1.*    # Collection-scoped index, catalog, and manifest
├── drift/
│   ├── coverage_check.py     # Part E semantic coverage & agglomerative clustering
│   └── user_queries.jsonl    # 40 live queries (28 in-domain, 12 out-of-domain)
├── eval/
│   ├── golden.jsonl          # 26 curated questions (13 human, 13 synthetic reviewed; dev/test splits)
│   ├── golden_v2.jsonl       # 32 updated questions post-Batch B drift
│   ├── human_labels_*.jsonl  # Per-candidate human-inspected answers for judge agreement
│   ├── gate_config.yaml      # Single source of truth for gate & threshold rules
│   ├── baseline.json         # Committed baseline benchmark reference
│   ├── run_eval.py           # Part B/C 3-level evaluation & gate runner
│   └── results/              # Canonical JSON & Markdown results for Parts A-E
├── k8s/                      # Numbered Minikube manifests, Kustomization & guide
├── scripts/                  # CLI entrypoints for Parts A, C, D, E & ingest Job
├── src/
│   ├── api/main.py           # FastAPI service (/healthz, /readyz, /metrics, /api/ask)
│   └── pipeline/             # config, embed, guardrails, verification, indexer, tracer, qa_engine
├── tests/                    # Automated unit & regression tests (pytest)
├── traces/                   # Trace inspection CLI & committed API + eval trace logs (280 total)
├── Dockerfile
├── pytest.ini
└── requirements.txt
```

---

## 3. Command Reference (Parts A to F & Tests)

### Prerequisites
```bash
git clone https://github.com/ali-akbar1214/RAHBAR.git
cd RAHBAR
pip install -r requirements.txt
# Add GROQ_API_KEY=... to local key.env (gitignored)
```

### Unit & Regression Tests
```bash
python -m pytest -v
```

### Part A: Data Verification, Incremental Ingestion, Archival & Rollback
```bash
python scripts/run_part_a.py
```

### Part B: Offline 3-Level Evaluation & Stochasticity Check
```bash
python eval/run_eval.py --save-baseline --require-llm
python eval/run_eval.py --stochasticity
```

### Part C: Change Experiment & Automated Release Gate
```bash
python scripts/release_gate.py --candidate all --require-llm
python scripts/release_gate.py --candidate chunk --require-llm
python scripts/release_gate.py --candidate embed --require-llm
python scripts/release_gate.py --candidate fail --require-llm
```

### Part D: Observability, Guardrails, Trace Lookup & Time-Windowed Metrics
```bash
python scripts/test_part_d.py
python -m traces.trace_tool --trace-id 29bfcd88-f00d-4116-851c-e9c8eb46f506
python scripts/trace_metrics.py --last 72h
```

### Part E: Coverage Monitoring, Agglomerative Clustering & Data Drift
```bash
python drift/coverage_check.py --label before --embedding-version lexical-proj-v2
python drift/coverage_check.py --golden-path eval/golden_v2.jsonl --label after --embedding-version lexical-proj-v2
python scripts/run_part_e_drift.py
```

### Part F: Kubernetes Cluster Deployment (Minikube)
```bash
docker build -t ragops:v1 .
minikube image load ragops:v1

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-pvc.yaml
kubectl apply -f k8s/20-configmap.yaml
kubectl create secret generic ragops-secret --from-env-file=key.env -n ragops

kubectl apply -f k8s/30-indexer-job.yaml
kubectl wait --for=condition=complete job/ragops-indexer-job -n ragops --timeout=120s

kubectl apply -f k8s/40-api-deployment.yaml
kubectl apply -f k8s/60-helper-pod.yaml
kubectl apply -f k8s/35-indexer-candidate-job.yaml
kubectl wait --for=condition=complete job/ragops-indexer-candidate-chunk -n ragops --timeout=120s

kubectl apply -f k8s/50-eval-job.yaml
kubectl logs -f job/ragops-eval-job -n ragops
```
See [k8s/README.md](k8s/README.md) for bad-candidate gating (`55-eval-bad-job.yaml`), ConfigMap gate mutation, rolling update/undo, and unseen batch ingestion steps.
