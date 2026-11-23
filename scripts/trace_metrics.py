"""Part D3 & F7: Time-Windowed Trace Telemetry Aggregator & Similarity Histogram."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import get_traces_dir
from src.pipeline.tracer import QueryTracer


def _parse_ts(ts_str: str) -> datetime.datetime | None:
    if not ts_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        try:
            dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _parse_last_duration(spec: str) -> timedelta:
    m = re.fullmatch(r"(\d+)\s*([smhd])", spec.strip().lower())
    if not m:
        raise ValueError(f"Invalid --last duration '{spec}' (expected e.g. 30m, 1h, 2d)")
    val, unit = int(m.group(1)), m.group(2)
    return {
        "s": timedelta(seconds=val),
        "m": timedelta(minutes=val),
        "h": timedelta(hours=val),
        "d": timedelta(days=val),
    }[unit]


def compute_metrics(
    traces_path_or_dir: str | Path | None = None,
    since: datetime.datetime | None = None,
    until: datetime.datetime | None = None,
) -> dict[str, Any]:
    target = Path(traces_path_or_dir) if traces_path_or_dir else get_traces_dir()
    files = QueryTracer.iter_trace_files(target)
    if not files:
        return {"error": f"No trace JSONL files found at {target}"}

    records: list[dict[str, Any]] = []
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            dt = _parse_ts(str(obj.get("timestamp", "")))
            if since and (dt is None or dt < since):
                continue
            if until and (dt is None or dt > until):
                continue
            records.append(obj)

    if not records:
        return {"error": "No trace records matched the requested time window."}

    total = len(records)
    cache_hits = sum(1 for r in records if r.get("cache_hit"))
    fallbacks = sum(1 for r in records if r.get("used_fallback"))
    abstentions = sum(
        1
        for r in records
        if any("abstain" in str(a) for a in r.get("guardrail_actions", []))
        or "do not have sufficient information" in str(r.get("response", "")).lower()
    )
    injections = sum(
        1
        for r in records
        if any(
            "rail" in str(a) or "injection" in str(a)
            for a in r.get("guardrail_actions", [])
        )
    )

    top1_scores: list[float] = []
    stage_latencies: dict[str, list[float]] = {}
    for r in records:
        chunks = r.get("retrieved_chunks") or []
        if chunks and "score" in chunks[0]:
            top1_scores.append(float(chunks[0]["score"]))
        for st, val in (r.get("latency_per_stage") or {}).items():
            stage_latencies.setdefault(st, []).append(float(val))

    top1_scores.sort()
    sim_dist = {}
    histogram: dict[str, int] = {
        "[0.00-0.20)": 0,
        "[0.20-0.30)": 0,
        "[0.30-0.40)": 0,
        "[0.40-0.60)": 0,
        "[0.60-1.00]": 0,
    }
    for s in top1_scores:
        if s < 0.20:
            histogram["[0.00-0.20)"] += 1
        elif s < 0.30:
            histogram["[0.20-0.30)"] += 1
        elif s < 0.40:
            histogram["[0.30-0.40)"] += 1
        elif s < 0.60:
            histogram["[0.40-0.60)"] += 1
        else:
            histogram["[0.60-1.00]"] += 1

    if top1_scores:
        sim_dist = {
            "min": round(top1_scores[0], 4),
            "median": round(top1_scores[len(top1_scores) // 2], 4),
            "max": round(top1_scores[-1], 4),
            "count": len(top1_scores),
        }

    p95_lat: dict[str, float] = {}
    for st, vals in stage_latencies.items():
        vals.sort()
        idx = min(len(vals) - 1, int(0.95 * len(vals)))
        p95_lat[st] = round(vals[idx], 3)

    return {
        "files_scanned": [p.name for p in files],
        "window_since": since.isoformat() if since else "all",
        "window_until": until.isoformat() if until else "now",
        "total_queries_logged": total,
        "cache_hit_rate": round(cache_hits / total, 4),
        "abstention_rate": round(abstentions / total, 4),
        "injection_attempts_flagged": injections,
        "fallback_count": fallbacks,
        "top1_similarity_distribution": sim_dist,
        "top1_similarity_histogram": histogram,
        "p95_latency_per_stage_ms": p95_lat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate time-windowed RAGOps trace metrics")
    parser.add_argument("--traces-file", default=None, help="Path to trace file or directory (default: TRACES_DIR)")
    parser.add_argument("--since", default=None, help="ISO-8601 start timestamp")
    parser.add_argument("--until", default=None, help="ISO-8601 end timestamp")
    parser.add_argument("--last", default=None, help="Relative window e.g. 1h, 30m, 2d")
    args = parser.parse_args()

    now = datetime.datetime.now(timezone.utc)
    since_dt = _parse_ts(args.since) if args.since else None
    until_dt = _parse_ts(args.until) if args.until else None
    if args.last:
        since_dt = now - _parse_last_duration(args.last)
        until_dt = now

    m = compute_metrics(args.traces_file, since=since_dt, until=until_dt)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
