"""Part D2: Trace Lookup & Prompt Reconstruction CLI (D-1, F-7).

Searches across all traces/*.jsonl files (or a specified file/directory) for a
given --trace-id and prints the full structured trace, prompt, and metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import get_traces_dir
from src.pipeline.tracer import QueryTracer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a RAGOps query trace by ID")
    parser.add_argument("--trace-id", required=True, help="UUID trace_id to inspect")
    parser.add_argument("--traces-path", default=None, help="Optional trace file or directory (defaults to TRACES_DIR)")
    args = parser.parse_args()

    t_path = Path(args.traces_path) if args.traces_path else get_traces_dir()
    tracer = QueryTracer(t_path if t_path.is_file() else t_path / "traces.jsonl", emit_stdout=False)
    rec = tracer.get_trace_by_id(args.trace_id)

    if rec is None:
        print(f"Trace ID '{args.trace_id}' not found under {t_path}.", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print(f"TRACE INSPECTION: {rec.trace_id}")
    print("=" * 78)
    print(f"Timestamp:         {rec.timestamp}")
    print(f"Collection:        {rec.collection_name}")
    print(f"Embedding:         {rec.embedding_model} ({rec.embedding_version})")
    print(f"Model & Params:    {rec.model} | {rec.parameters}")
    print(f"Cache Hit:         {rec.cache_hit} | Used Fallback: {rec.used_fallback}")
    print(f"Filters Applied:   {rec.filters_applied}")
    print(f"Guardrail Actions: {rec.guardrail_actions}")
    print(f"Latency (ms):      {rec.latency_per_stage}")
    print(f"User Query:        {rec.user_query}")
    print(f"Rewritten Query:   {rec.rewritten_query}")
    print(f"Reranked Order:    {rec.reranked_order}")
    print("\n--- Retrieved Chunks ---")
    for c in rec.retrieved_chunks:
        print(
            f"  - {c.get('chunk_id')} (doc={c.get('doc_id')}, ver={c.get('doc_version')}, "
            f"page={c.get('page')}, score={c.get('score')}, sec={c.get('section')})"
        )
    print(f"\n--- Prompt (sha256={rec.prompt_hash}) ---")
    print(rec.full_prompt)
    print("\n--- Response & Citations ---")
    print(rec.response)
    print(f"Citations: {rec.citations}")
    print("=" * 78)


if __name__ == "__main__":
    main()
