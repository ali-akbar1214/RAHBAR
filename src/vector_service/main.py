"""Dedicated Vector-Store Microservice (Part F1).

Serves read-only collection JSON files from ragops-data-pvc over HTTP so the
API pods stay stateless regarding vector storage while ragops-indexer-job retains
sole-writer ownership of /data/index.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.config import get_collection_name, get_index_path, get_manifest_path
from src.pipeline.indexer import VectorStore

app = FastAPI(
    title="RAHBAR Vector Store Service",
    description="Collection-scoped vector similarity lookup service over PVC storage",
    version="1.0.0",
)

_stores: dict[str, VectorStore] = {}


def _get_store(collection: str) -> VectorStore:
    path = get_index_path(collection)
    store = _stores.get(collection)
    if store is None or store.storage_path != path:
        store = VectorStore(path)
        _stores[collection] = store
    else:
        store.reload_if_changed()
    return store


def _load_manifest(collection: str) -> dict[str, Any] | None:
    mpath = get_manifest_path(collection)
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


class QueryRequest(BaseModel):
    collection: str | None = None
    vector: list[float] = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)
    filter_superseded: bool = True


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(collection: str | None = Query(default=None)) -> dict[str, Any]:
    col = collection or get_collection_name()
    store = _get_store(col)
    active = store.total_chunks(include_superseded=False)
    manifest = _load_manifest(col)
    if active == 0:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "collection": col, "active_chunks": 0},
        )
    return {
        "status": "ready",
        "collection": col,
        "active_chunks": active,
        "index_path": str(store.storage_path),
        "manifest": manifest or {},
    }


@app.post("/query")
def query_vectors(req: QueryRequest) -> dict[str, Any]:
    col = req.collection or get_collection_name()
    store = _get_store(col)
    hits = store.query(
        req.vector,
        top_k=req.top_k,
        filter_superseded=req.filter_superseded,
    )
    results = []
    for chunk, score in hits:
        payload = asdict(chunk)
        # Omit large dense vectors over the wire; QAEngine only needs metadata + text
        payload["vector"] = []
        results.append({"chunk": payload, "score": round(float(score), 6)})
    return {"collection": col, "results": results}
