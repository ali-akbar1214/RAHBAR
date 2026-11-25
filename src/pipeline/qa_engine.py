"""End-to-End Grounded Q&A Service with Tracing, Semantic Caching, and Dual Guardrails.

Pipeline stages:
  1. mtime index reload check (F-4) & collection manifest verification (F-3).
  2. Query normalization / rewriting (D-1).
  3. Semantic Cache check (collection + embedding_version scoped, cosine >= 0.95) (D-1).
  4. Input Guardrail Rail via shared guardrails.py (A-6 / D-4).
  5. Filtered Vector Retrieval + Lexical Section-Overlap Reranking (D-1).
  6. Retrieval Guardrail Rail on retrieved chunks before prompt assembly (D-4).
  7. Prompt Construction & SHA-256 Hashing.
  8. Groq LLM Generation (temperature=0, seed=42) with explicit `used_fallback` tracking (B-8).
  9. Full bare JSONL Trace Output (D-2 / F-7).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import time
import uuid
from datetime import timezone
from pathlib import Path
from typing import Any

import httpx

from .config import (
    get_collection_name,
    get_embedding_version,
    get_index_dir,
    get_index_path,
    get_manifest_path,
    get_relevance_threshold,
)
from .embed import DEFAULT_EMBEDDING_VERSION, cosine_similarity, embed_text
from .guardrails import scan_text_for_injection
from .indexer import ChunkRecord, VectorStore
from .tracer import QueryTracer, TraceRecord

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
SEMANTIC_CACHE_THRESHOLD = 0.95


def normalize_and_rewrite_query(query: str) -> str:
    """Normalize whitespace/case and expand common FAST-NUCES policy abbreviations."""
    q = re.sub(r"\s+", " ", query.strip())
    expansions = {
        r"\bcgpa\b": "CGPA cumulative grade point average",
        r"\bsgpa\b": "SGPA semester grade point average",
        r"\bhod\b": "HOD head of department",
        r"\bfa\b": "FA failure due to attendance",
        r"\bfydp\b": "FYDP final year design project",
    }
    rewritten = q
    for pat, rep in expansions.items():
        if re.search(pat, rewritten, re.IGNORECASE):
            rewritten = re.sub(pat, rep, rewritten, flags=re.IGNORECASE)
    return rewritten


def rerank_chunks(query: str, retrieved: list[tuple[ChunkRecord, float]], top_k: int = 3) -> list[tuple[ChunkRecord, float]]:
    """Lexical section-title overlap reranker over top retrieved candidates."""
    q_terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    scored: list[tuple[ChunkRecord, float, float]] = []
    for chunk, base_sim in retrieved:
        sec_terms = set(re.findall(r"[a-z0-9]{3,}", f"{chunk.section} {chunk.title}".lower()))
        overlap = len(q_terms & sec_terms)
        boost = min(0.06, 0.015 * overlap)
        scored.append((chunk, base_sim + boost, base_sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(c, round(orig_sim, 6)) for c, _, orig_sim in scored[:top_k]]


class QAEngine:
    def __init__(
        self,
        vector_store_path: str | Path | None = None,
        traces_path: str | Path | None = None,
        groq_api_key: str | None = None,
        model_name: str = DEFAULT_GROQ_MODEL,
        collection_name: str | None = None,
        embedding_version: str | None = None,
        relevance_threshold: float | None = None,
        emit_stdout_traces: bool = True,
        vector_store_url: str | None = None,
    ):
        self.collection_name = collection_name or get_collection_name()
        self.embedding_version = embedding_version or get_embedding_version() or DEFAULT_EMBEDDING_VERSION
        self.vector_store_url = (
            vector_store_url
            if vector_store_url is not None
            else os.environ.get("VECTOR_STORE_URL", "")
        ).strip().rstrip("/")
        self.index_path = Path(vector_store_path) if vector_store_path else get_index_path(self.collection_name)
        self.manifest_path = self._resolve_manifest_path(self.index_path)

        # When VECTOR_STORE_URL is configured, query the dedicated vector-store service over HTTP.
        self.vector_store: VectorStore | None = None if self.vector_store_url else VectorStore(self.index_path)
        self.tracer = QueryTracer(traces_path, emit_stdout=emit_stdout_traces)
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
        self.model_name = os.environ.get("LLM_MODEL", model_name)
        self.relevance_threshold = (
            relevance_threshold if relevance_threshold is not None else get_relevance_threshold()
        )

        # Cache keyed by (collection_name, embedding_version) -> list of (q_vec, q_norm, payload)
        self.cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

        self.groq_client = None
        if self.groq_api_key and not self.groq_api_key.startswith("REPLACE"):
            try:
                from groq import Groq

                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print(f"[QAEngine] Warning: Could not initialize Groq ({e})")

    @staticmethod
    def _resolve_manifest_path(index_path: Path) -> Path:
        if index_path.name.endswith(".json"):
            return index_path.with_name(index_path.stem + ".manifest.json")
        return get_manifest_path()

    def get_manifest(self) -> dict[str, Any] | None:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def check_readiness(self) -> tuple[bool, dict[str, Any]]:
        """Verify store is non-empty and manifest embedding version matches engine config (F-3, F-4)."""
        info: dict[str, Any] = {
            "collection": self.collection_name,
            "embedding_version": self.embedding_version,
        }
        if self.vector_store_url:
            info["vector_store_url"] = self.vector_store_url
            try:
                r = httpx.get(
                    f"{self.vector_store_url}/readyz",
                    params={"collection": self.collection_name},
                    timeout=5.0,
                )
                if r.status_code != 200:
                    info["reason"] = f"Vector store service not ready ({r.status_code})"
                    return False, info
                data = r.json()
                chunk_count = int(data.get("active_chunks", 0))
                manifest = data.get("manifest") or {}
                info["active_chunks"] = chunk_count
                info["index_path"] = data.get("index_path", str(self.index_path))
                if chunk_count == 0:
                    info["reason"] = "Vector store has 0 active chunks"
                    return False, info
                if manifest.get("embedding_version") and manifest["embedding_version"] != self.embedding_version:
                    info["manifest_embedding_version"] = manifest["embedding_version"]
                    info["reason"] = (
                        f"Embedding version mismatch: API configured for '{self.embedding_version}', "
                        f"manifest has '{manifest['embedding_version']}'"
                    )
                    return False, info
                return True, info
            except Exception as e:
                info["reason"] = f"Vector store service unreachable: {e}"
                return False, info

        assert self.vector_store is not None
        self.vector_store.reload_if_changed()
        manifest = self.get_manifest()
        chunk_count = self.vector_store.total_chunks(include_superseded=False)
        info["active_chunks"] = chunk_count
        info["index_path"] = str(self.index_path)
        if chunk_count == 0:
            info["reason"] = "Vector store has 0 active chunks"
            return False, info
        if manifest and manifest.get("embedding_version") and manifest["embedding_version"] != self.embedding_version:
            info["manifest_embedding_version"] = manifest["embedding_version"]
            info["reason"] = (
                f"Embedding version mismatch: API configured for '{self.embedding_version}', "
                f"manifest has '{manifest['embedding_version']}'"
            )
            return False, info
        return True, info

    def _query_vectors(self, collection: str, q_vec: list[float], top_k: int) -> list[tuple[ChunkRecord, float]]:
        """Query either the dedicated vector-store service or local VectorStore."""
        if self.vector_store_url:
            resp = httpx.post(
                f"{self.vector_store_url}/query",
                json={
                    "collection": collection,
                    "vector": q_vec,
                    "top_k": top_k,
                    "filter_superseded": True,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            rows = resp.json().get("results", [])
            return [(ChunkRecord(**row["chunk"]), float(row["score"])) for row in rows]
        assert self.vector_store is not None
        return self.vector_store.query(q_vec, top_k=top_k, filter_superseded=True)

    def _lookup_cache(
        self, cache_scope: tuple[str, str], q_norm: str, q_vec: list[float]
    ) -> dict[str, Any] | None:
        entries = self.cache.get(cache_scope, [])
        for entry in entries:
            if entry["q_norm"] == q_norm:
                return entry["payload"]
            if cosine_similarity(q_vec, entry["q_vec"]) >= SEMANTIC_CACHE_THRESHOLD:
                return entry["payload"]
        return None

    def answer_query(
        self,
        user_query: str,
        collection_name: str | None = None,
        embedding_version: str | None = None,
        top_k: int = 3,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        if self.vector_store is not None:
            self.vector_store.reload_if_changed()

        col = collection_name or self.collection_name
        emb_ver = embedding_version or self.embedding_version
        manifest = self.get_manifest()
        emb_model = (manifest or {}).get("embedding_model", emb_ver)

        trace_id = str(uuid.uuid4())
        latencies: dict[str, float] = {}
        guardrail_actions: list[str] = []
        ts_iso = datetime.datetime.now(timezone.utc).isoformat()

        rewritten_query = normalize_and_rewrite_query(user_query)

        # 1. Input Guardrail Check (Prompt Injection Defense)
        t_g0 = time.perf_counter()
        inj_pat = scan_text_for_injection(user_query)
        latencies["guardrail_ms"] = round((time.perf_counter() - t_g0) * 1000.0, 3)

        if inj_pat:
            guardrail_actions.append(f"input_rail: blocked '{inj_pat}'")
            latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
            abstain_resp = "I cannot fulfill this request. Security guardrail blocked potential prompt injection."
            rec = TraceRecord(
                trace_id=trace_id,
                timestamp=ts_iso,
                user_query=user_query,
                rewritten_query=rewritten_query,
                embedding_model=emb_model,
                embedding_version=emb_ver,
                collection_name=col,
                retrieved_chunks=[],
                filters_applied={"superseded": False},
                reranked_order=[],
                prompt_hash=hashlib.sha256(user_query.encode("utf-8")).hexdigest()[:16],
                full_prompt=user_query,
                model=self.model_name,
                parameters={"temperature": 0.0, "max_tokens": 300, "seed": 42},
                response=abstain_resp,
                citations=[],
                guardrail_actions=guardrail_actions,
                latency_per_stage=latencies,
                cache_hit=False,
                used_fallback=False,
            )
            self.tracer.record_trace(rec)
            return {
                "trace_id": trace_id,
                "response": abstain_resp,
                "citations": [],
                "abstain": True,
                "cache_hit": False,
                "used_fallback": False,
                "latency_ms": latencies["total_ms"],
                "guardrail_actions": guardrail_actions,
            }

        # 2. Embedding + Semantic Cache Check (collection + embedding_version scoped)
        t_c0 = time.perf_counter()
        q_vec = embed_text(rewritten_query, version=emb_ver)
        q_norm = user_query.strip().lower()
        cache_scope = (col, emb_ver)
        cached_payload = self._lookup_cache(cache_scope, q_norm, q_vec) if use_cache else None
        latencies["cache_ms"] = round((time.perf_counter() - t_c0) * 1000.0, 3)

        if cached_payload is not None:
            latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
            rec = TraceRecord(
                trace_id=trace_id,
                timestamp=ts_iso,
                user_query=user_query,
                rewritten_query=rewritten_query,
                embedding_model=emb_model,
                embedding_version=emb_ver,
                collection_name=col,
                retrieved_chunks=cached_payload["retrieved_chunks"],
                filters_applied={"superseded": False},
                reranked_order=[c["chunk_id"] for c in cached_payload["retrieved_chunks"]],
                prompt_hash=cached_payload["prompt_hash"],
                full_prompt=cached_payload["full_prompt"],
                model=self.model_name,
                parameters={"temperature": 0.0, "max_tokens": 300, "seed": 42},
                response=cached_payload["response"],
                citations=cached_payload["citations"],
                guardrail_actions=["cache_hit"],
                latency_per_stage=latencies,
                cache_hit=True,
                used_fallback=cached_payload.get("used_fallback", False),
            )
            self.tracer.record_trace(rec)
            return {
                "trace_id": trace_id,
                "response": cached_payload["response"],
                "citations": cached_payload["citations"],
                "abstain": cached_payload["abstain"],
                "cache_hit": True,
                "used_fallback": cached_payload.get("used_fallback", False),
                "latency_ms": latencies["total_ms"],
                "retrieved_chunks": cached_payload["retrieved_chunks"],
                "guardrail_actions": ["cache_hit"],
            }

        # 3. Vector Retrieval + Lexical Reranking
        t_r0 = time.perf_counter()
        candidates = self._query_vectors(col, q_vec, top_k=max(6, top_k * 2))
        retrieved_reranked = rerank_chunks(user_query, candidates, top_k=top_k)
        latencies["retrieval_ms"] = round((time.perf_counter() - t_r0) * 1000.0, 3)

        top_score = retrieved_reranked[0][1] if retrieved_reranked else 0.0

        # 4. Out-of-domain Abstention Check
        if top_score < self.relevance_threshold:
            guardrail_actions.append(
                f"out_of_domain_abstain (top_score={top_score:.4f} < {self.relevance_threshold})"
            )
            latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
            abstain_msg = "I do not have sufficient information in the verified university policies to answer this question."
            chunk_summaries = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_version": c.doc_version,
                    "page": c.page,
                    "score": round(score, 4),
                    "section": c.section,
                }
                for c, score in retrieved_reranked
            ]
            rec = TraceRecord(
                trace_id=trace_id,
                timestamp=ts_iso,
                user_query=user_query,
                rewritten_query=rewritten_query,
                embedding_model=emb_model,
                embedding_version=emb_ver,
                collection_name=col,
                retrieved_chunks=chunk_summaries,
                filters_applied={"superseded": False},
                reranked_order=[c.chunk_id for c, _ in retrieved_reranked],
                prompt_hash=hashlib.sha256(user_query.encode("utf-8")).hexdigest()[:16],
                full_prompt=user_query,
                model=self.model_name,
                parameters={"temperature": 0.0, "max_tokens": 300, "seed": 42},
                response=abstain_msg,
                citations=[],
                guardrail_actions=guardrail_actions,
                latency_per_stage=latencies,
                cache_hit=False,
                used_fallback=False,
            )
            self.tracer.record_trace(rec)
            return {
                "trace_id": trace_id,
                "response": abstain_msg,
                "citations": [],
                "abstain": True,
                "cache_hit": False,
                "used_fallback": False,
                "latency_ms": latencies["total_ms"],
                "retrieved_chunks": chunk_summaries,
                "guardrail_actions": guardrail_actions,
            }

        # 5. Retrieval Guardrail Rail (scan retrieved chunks before prompt assembly, D-4)
        clean_chunks: list[tuple[ChunkRecord, float]] = []
        for c, score in retrieved_reranked:
            chunk_inj = scan_text_for_injection(c.text)
            if chunk_inj:
                guardrail_actions.append(f"retrieval_rail: dropped {c.chunk_id} ({chunk_inj})")
            else:
                clean_chunks.append((c, score))

        if not clean_chunks:
            guardrail_actions.append("retrieval_rail: all retrieved chunks dropped -> abstain")
            latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
            abstain_msg = "Retrieved context contained unsafe instructions and was blocked by security guardrails."
            rec = TraceRecord(
                trace_id=trace_id,
                timestamp=ts_iso,
                user_query=user_query,
                rewritten_query=rewritten_query,
                embedding_model=emb_model,
                embedding_version=emb_ver,
                collection_name=col,
                retrieved_chunks=[],
                filters_applied={"superseded": False},
                reranked_order=[],
                prompt_hash=hashlib.sha256(user_query.encode("utf-8")).hexdigest()[:16],
                full_prompt=user_query,
                model=self.model_name,
                parameters={"temperature": 0.0, "max_tokens": 300, "seed": 42},
                response=abstain_msg,
                citations=[],
                guardrail_actions=guardrail_actions,
                latency_per_stage=latencies,
                cache_hit=False,
                used_fallback=False,
            )
            self.tracer.record_trace(rec)
            return {
                "trace_id": trace_id,
                "response": abstain_msg,
                "citations": [],
                "abstain": True,
                "cache_hit": False,
                "used_fallback": False,
                "latency_ms": latencies["total_ms"],
                "retrieved_chunks": [],
                "guardrail_actions": guardrail_actions,
            }

        # 6. Prompt Assembly
        context_blocks = []
        citations = []
        retrieved_chunks_info = []
        for c, score in clean_chunks:
            context_blocks.append(f"[{c.source} | {c.section} | page {c.page}]:\n{c.text}")
            citations.append(
                {
                    "source": c.source,
                    "section": c.section,
                    "doc_version": c.doc_version,
                    "page": c.page,
                }
            )
            retrieved_chunks_info.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_version": c.doc_version,
                    "page": c.page,
                    "score": round(score, 4),
                    "section": c.section,
                    "text_snippet": c.text[:160] + ("..." if len(c.text) > 160 else ""),
                }
            )

        joined_context = "\n\n".join(context_blocks)
        system_instruction = (
            "You are RAHBAR, an official academic policy assistant for FAST-NUCES. "
            "Answer the user question concisely and accurately using ONLY the provided context. "
            "If the context does not contain the answer, reply strictly with: "
            "'I do not have sufficient information in the verified university policies to answer this question.' "
            "Always cite the source filename and section."
        )
        full_prompt = (
            f"System: {system_instruction}\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Question: {user_query}\n\nAnswer:"
        )
        prompt_hash = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()

        # 7. Generation Stage (Groq LLM)
        t_gen0 = time.perf_counter()
        answer_text = ""
        used_fallback = False
        llm_error: str | None = None

        if self.groq_client:
            try:
                resp = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Context:\n{joined_context}\n\nQuestion: {user_query}"},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                    seed=42,
                )
                answer_text = (resp.choices[0].message.content or "").strip()
                if not answer_text:
                    raise RuntimeError("LLM returned empty content")
            except Exception as e:
                used_fallback = True
                llm_error = str(e)
                guardrail_actions.append(f"llm_fallback: {llm_error[:120]}")
                answer_text = (
                    f"According to [{citations[0]['source']} / {citations[0]['section']}]: "
                    f"{clean_chunks[0][0].text[:250]}"
                )
        else:
            used_fallback = True
            llm_error = "Groq client not initialized (missing GROQ_API_KEY)"
            guardrail_actions.append("llm_fallback: missing_api_key")
            answer_text = (
                f"According to [{citations[0]['source']} / {citations[0]['section']}]: "
                f"{clean_chunks[0][0].text[:250]}"
            )

        latencies["generation_ms"] = round((time.perf_counter() - t_gen0) * 1000.0, 3)
        latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)

        abstain_flag = "do not have sufficient information" in answer_text.lower()

        result = {
            "trace_id": trace_id,
            "response": answer_text,
            "citations": [] if abstain_flag else citations,
            "abstain": abstain_flag,
            "cache_hit": False,
            "used_fallback": used_fallback,
            "llm_error": llm_error,
            "latency_ms": latencies["total_ms"],
            "retrieved_chunks": retrieved_chunks_info,
            "prompt_hash": prompt_hash,
            "full_prompt": full_prompt,
            "guardrail_actions": guardrail_actions or ["none"],
        }

        if use_cache and not used_fallback:
            self.cache.setdefault(cache_scope, []).append(
                {
                    "q_norm": q_norm,
                    "q_vec": q_vec,
                    "payload": result,
                }
            )

        rec = TraceRecord(
            trace_id=trace_id,
            timestamp=ts_iso,
            user_query=user_query,
            rewritten_query=rewritten_query,
            embedding_model=emb_model,
            embedding_version=emb_ver,
            collection_name=col,
            retrieved_chunks=retrieved_chunks_info,
            filters_applied={"superseded": False},
            reranked_order=[c["chunk_id"] for c in retrieved_chunks_info],
            prompt_hash=prompt_hash,
            full_prompt=full_prompt,
            model=self.model_name,
            parameters={"temperature": 0.0, "max_tokens": 300, "seed": 42},
            response=answer_text,
            citations=result["citations"],
            guardrail_actions=guardrail_actions or ["none"],
            latency_per_stage=latencies,
            cache_hit=False,
            used_fallback=used_fallback,
        )
        self.tracer.record_trace(rec)
        return result
