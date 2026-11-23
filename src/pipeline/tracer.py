"""Part D1 & D2: Structured Query Tracing & Per-Pod JSONL Logger.

- Emits bare JSON objects to stdout with "event": "trace" so `kubectl logs ... | jq` works (D-2).
- Writes per-pod trace files (traces-<HOSTNAME>.jsonl) in TRACES_DIR to avoid RWO lock contention (F-7).
- Supports reading across all traces/*.jsonl files.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import get_traces_dir


@dataclass
class TraceRecord:
    trace_id: str
    timestamp: str
    user_query: str
    rewritten_query: str | None
    embedding_model: str
    embedding_version: str
    collection_name: str
    retrieved_chunks: list[dict[str, Any]]
    filters_applied: dict[str, Any]
    reranked_order: list[str]
    prompt_hash: str
    full_prompt: str
    model: str
    parameters: dict[str, Any]
    response: str
    citations: list[dict[str, Any]]
    guardrail_actions: list[str]
    latency_per_stage: dict[str, float]
    cache_hit: bool
    used_fallback: bool = False
    event: str = "trace"

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


def default_trace_file(traces_dir: str | Path | None = None) -> Path:
    t_dir = Path(traces_dir) if traces_dir else get_traces_dir()
    host = os.environ.get("HOSTNAME")
    if host:
        safe_host = "".join(c if c.isalnum() or c in "-_." else "_" for c in host)
        return t_dir / f"traces-{safe_host}.jsonl"
    return t_dir / "traces.jsonl"


class QueryTracer:
    """Manages appending bare JSONL traces and querying across traces/*.jsonl."""

    def __init__(self, log_path: str | Path | None = None, emit_stdout: bool = True):
        self.log_path = Path(log_path) if log_path else default_trace_file()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.emit_stdout = emit_stdout

    def record_trace(self, record: TraceRecord) -> None:
        json_line = record.to_json_line()
        if self.emit_stdout:
            print(json_line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

    @staticmethod
    def iter_trace_files(path_or_dir: str | Path | None = None) -> list[Path]:
        target = Path(path_or_dir) if path_or_dir else get_traces_dir()
        if target.is_file():
            return [target]
        if target.is_dir():
            return sorted(target.glob("*.jsonl"))
        return []

    def get_trace_by_id(self, trace_id: str) -> TraceRecord | None:
        # Search own log_path first, then all jsonl files in parent dir
        candidates = [self.log_path] if self.log_path.exists() else []
        for p in self.iter_trace_files(self.log_path.parent):
            if p not in candidates:
                candidates.append(p)
        for p in candidates:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("trace_id") == trace_id:
                        return TraceRecord(**{k: v for k, v in data.items() if k in TraceRecord.__dataclass_fields__})
        return None

    def get_all_traces(self) -> list[TraceRecord]:
        traces: list[TraceRecord] = []
        files = self.iter_trace_files(self.log_path.parent if self.log_path.parent.exists() else None)
        if not files and self.log_path.exists():
            files = [self.log_path]
        for p in files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        traces.append(
                            TraceRecord(**{k: v for k, v in data.items() if k in TraceRecord.__dataclass_fields__})
                        )
        return traces
