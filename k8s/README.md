# Kubernetes Deployment & Operations Guide (Part F)

## 1. Architecture & Sole-Writer Enforcement
- **`ragops-data-pvc` (1Gi RWO)**: Holds `/data/lake/{incoming,raw,quarantine,archive}` and `/data/index/{COLLECTION_NAME}.json` + `.manifest.json`.
- **Sole-Writer & Dedicated Vector-Store Architecture**: `ragops-data-pvc` is mounted **read-write ONLY** in `ragops-indexer-job` and `ragops-indexer-cronjob` (`scripts/ingest_job.py`). A dedicated **vector-store microservice** (`38-vector-store.yaml`: `Deployment/ragops-vector` + `Service/ragops-vector` on `:8001`) mounts `ragops-data-pvc` `readOnly: true` and serves `/readyz` + `/query`. Both `ragops-api` and `ragops-api-candidate` query it over `VECTOR_STORE_URL=http://ragops-vector:8001` without mounting the data PVC directly; `ragops-eval-job` mounts `ragops-data-pvc` `readOnly: true` only for ground-truth chunk lookup during LLM judging.
- **`ragops-results-pvc` (500Mi RWO)**: Persists verification, ingestion, evaluation, and gate JSON reports.
- **`ragops-traces-pvc` (500Mi RWO)**: Each API pod writes `traces-$HOSTNAME.jsonl` to avoid concurrent write contention.

---

## 2. Clean-Cluster Deployment Order

### Step 1: Build & Load Single Docker Image into Minikube
```bash
docker build -t ragops:v1 .
minikube image load ragops:v1
```

### Step 2: Apply Namespace, PVCs, ConfigMaps, and Create Secret
```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-pvc.yaml
kubectl apply -f k8s/20-configmap.yaml
kubectl create secret generic ragops-secret --from-env-file=key.env -n ragops
```

### Step 3: Run Baseline Ingestion Job (Seeds Lake + Builds `fast-policies-v1`)
```bash
kubectl apply -f k8s/30-indexer-job.yaml
kubectl wait --for=condition=complete job/ragops-indexer-job -n ragops --timeout=120s
kubectl logs job/ragops-indexer-job -n ragops
```

### Step 4: Deploy Vector-Store Service, Live APIs & Helper Pod
```bash
kubectl apply -f k8s/38-vector-store.yaml
kubectl rollout status deployment/ragops-vector -n ragops
kubectl apply -f k8s/40-api-deployment.yaml
kubectl apply -f k8s/45-api-candidate-deployment.yaml
kubectl apply -f k8s/60-helper-pod.yaml
kubectl rollout status deployment/ragops-api -n ragops
kubectl rollout status deployment/ragops-api-candidate -n ragops
```

### Step 5: Build Candidate Collections & Run Gated Evaluation Jobs
```bash
# Passing single-change candidate (fast-policies-chunk, chunk 500/100, lexical-proj-v1)
kubectl apply -f k8s/35-indexer-candidate-job.yaml
kubectl wait --for=condition=complete job/ragops-indexer-candidate-chunk -n ragops --timeout=120s
# Eval Job targets live http://ragops-api-candidate:8000 via EVAL_API_URL (no hard-coded CLI flags)
kubectl apply -f k8s/50-eval-job.yaml
kubectl wait --for=condition=complete job/ragops-eval-job -n ragops --timeout=240s
kubectl logs job/ragops-eval-job -n ragops

# Failing bad candidate (fast-policies-bad, chunk_size=40, overlap=5) — stopped by gate (exit 1)
kubectl apply -f k8s/36-indexer-bad-job.yaml
kubectl wait --for=condition=complete job/ragops-indexer-bad -n ragops --timeout=120s
kubectl apply -f k8s/55-eval-bad-job.yaml
kubectl logs -f job/ragops-eval-bad-job -n ragops
```

### Step 6: ConfigMap Gate Mutation Without Image Rebuild
```bash
kubectl patch configmap ragops-gate-config -n ragops --type merge -p '{"data":{"min_faithfulness":"0.99"}}'
kubectl delete job ragops-eval-job -n ragops
kubectl apply -f k8s/50-eval-job.yaml
# Job now exits non-zero (FAIL) without rebuilding ragops:v1
kubectl patch configmap ragops-gate-config -n ragops --type merge -p '{"data":{"min_faithfulness":"0.80"}}'
```

### Step 7: Bad-Release Deployment & Rollback (`kubectl rollout undo`)
```bash
# Deploy a bad configuration (fast-policies-bad 40/5 chunks -> fails/abstains on /api/ask)
kubectl set env deployment/ragops-api COLLECTION_NAME=fast-policies-bad CHUNK_SIZE=40 CHUNK_OVERLAP=5 RELEVANCE_THRESHOLD=0.263 -n ragops
kubectl rollout status deployment/ragops-api -n ragops

# Rollback bad release to verified baseline fast-policies-v1
kubectl rollout undo deployment/ragops-api -n ragops
kubectl rollout status deployment/ragops-api -n ragops
```

### Step 8: Ingest Unseen Test Batch via Helper Pod
```bash
kubectl cp corpus/batch_b/. ragops/ragops-helper:/data/lake/incoming/
kubectl delete job ragops-indexer-job -n ragops
kubectl apply -f k8s/30-indexer-job.yaml
kubectl logs -f job/ragops-indexer-job -n ragops
```
