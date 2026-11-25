"""FastAPI Production Service for RAGOps Pipeline.

Endpoints:
  - GET  /healthz : Kubernetes liveness probe.
  - GET  /readyz  : Kubernetes readiness probe (validates index, chunks, & manifest version).
  - GET  /metrics : Prometheus-formatted operational telemetry across traces/*.jsonl.
  - POST /api/ask : Grounded Q&A endpoint with tracing, caching, and dual guardrails.
"""

from __future__ import annotations

import datetime
import sys
import time
from datetime import timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.trace_metrics import compute_metrics
from src.pipeline.config import (
    get_collection_name,
    get_embedding_version,
    get_index_path,
    get_traces_dir,
)
from src.pipeline.qa_engine import QAEngine
from src.pipeline.tracer import default_trace_file

app = FastAPI(
    title="RAHBAR RAGOps API Service",
    description="Grounded Question Answering Service over FAST-NUCES Regulations",
    version="2.1.0",
)

COLLECTION_NAME = get_collection_name()
EMBEDDING_VERSION = get_embedding_version()
INDEX_PATH = get_index_path(COLLECTION_NAME)
TRACES_DIR = get_traces_dir()

engine = QAEngine(
    vector_store_path=INDEX_PATH,
    traces_path=default_trace_file(TRACES_DIR),
    collection_name=COLLECTION_NAME,
    embedding_version=EMBEDDING_VERSION,
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Student question")
    use_cache: bool = Field(default=True, description="Enable semantic cache lookup")


class AskResponse(BaseModel):
    trace_id: str
    response: str
    citations: list[dict[str, Any]]
    abstain: bool
    cache_hit: bool = False
    used_fallback: bool = False
    latency_total_ms: float = 0.0
    retrieved_chunks: list[dict[str, Any]] = []


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe: verifies process is alive without hitting downstream stores."""
    return {"status": "ok", "timestamp": datetime.datetime.now(timezone.utc).isoformat()}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    """Readiness probe: validates index existence, active chunks, and manifest version."""
    ready, info = engine.check_readiness()
    if not ready:
        raise HTTPException(status_code=503, detail=info)
    return {"status": "ready", **info}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-compatible operational metrics endpoint."""
    m = compute_metrics(TRACES_DIR)
    if "error" in m:
        body = (
            "# HELP ragops_queries_total Total number of queries received\n"
            "# TYPE ragops_queries_total counter\n"
            "ragops_queries_total 0\n"
        )
        return Response(content=body, media_type="text/plain")

    lines = [
        "# HELP ragops_queries_total Total number of queries processed",
        "# TYPE ragops_queries_total counter",
        f"ragops_queries_total {m.get('total_queries_logged', 0)}",
        "# HELP ragops_cache_hit_ratio Semantic cache hit ratio",
        "# TYPE ragops_cache_hit_ratio gauge",
        f"ragops_cache_hit_ratio {m.get('cache_hit_rate', 0.0)}",
        "# HELP ragops_abstention_ratio Out-of-scope abstention ratio",
        "# TYPE ragops_abstention_ratio gauge",
        f"ragops_abstention_ratio {m.get('abstention_rate', 0.0)}",
        "# HELP ragops_injections_flagged_total Total prompt injections blocked",
        "# TYPE ragops_injections_flagged_total counter",
        f"ragops_injections_flagged_total {m.get('injection_attempts_flagged', 0)}",
        "# HELP ragops_top1_similarity_median Median top-1 retrieval cosine similarity",
        "# TYPE ragops_top1_similarity_median gauge",
        f"ragops_top1_similarity_median {m.get('top1_similarity_distribution', {}).get('median', 0.0)}",
    ]

    p95 = m.get("p95_latency_per_stage_ms", {})
    for stage, val in p95.items():
        lines.extend([
            f"# HELP ragops_latency_p95_ms_{stage} p95 latency for {stage} in milliseconds",
            f"# TYPE ragops_latency_p95_ms_{stage} gauge",
            f"ragops_latency_p95_ms_{stage} {val}",
        ])

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    res = engine.answer_query(user_query=req.query, use_cache=req.use_cache)
    total_lat = round((time.perf_counter() - t0) * 1000.0, 3)

    return AskResponse(
        trace_id=res["trace_id"],
        response=res["response"],
        citations=res["citations"],
        abstain=res["abstain"],
        cache_hit=res.get("cache_hit", False),
        used_fallback=res.get("used_fallback", False),
        latency_total_ms=total_lat,
        retrieved_chunks=res.get("retrieved_chunks", []),
    )
