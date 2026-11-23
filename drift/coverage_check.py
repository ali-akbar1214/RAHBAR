"""Part E: Live Query Coverage, Drift Monitoring & Agglomerative Clustering (E1 - E4).

- Derives similarity threshold from Part B module-level distribution (or --threshold).
- Reports best-match score distributions separately for in_domain vs out_of_domain queries.
- Lists EVERY uncovered query with its nearest golden question and score.
- Groups ALL uncovered queries via pure-NumPy agglomerative clustering on embeddings,
  labelling each cluster by its top frequent terms (zero unclustered queries).
- Writes structured JSON report to eval/results/coverage_<label>_<ts>.json.
- Exits with code 1 when coverage < 85% target, and 0 when >= 85%.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import PROJECT_ROOT, get_embedding_version, get_results_dir
from src.pipeline.embed import cosine_similarity, embed_text

TARGET_COVERAGE_SHARE = 0.85

_STOPWORDS = {
    "what", "where", "when", "which", "who", "whom", "how", "is", "are", "the",
    "for", "and", "or", "to", "in", "on", "of", "at", "do", "does", "can", "i",
    "my", "me", "a", "an", "be", "with", "from", "there", "any", "if", "you",
    "your", "about", "get", "take", "much", "many",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_threshold_from_part_b(embedding_version: str) -> tuple[float, str]:
    """Derive coverage threshold from Part B paraphrase vs unrelated distribution."""
    base_path = PROJECT_ROOT / "eval" / "baseline.json"
    if base_path.exists():
        b = json.loads(base_path.read_text(encoding="utf-8"))
        m = b.get("module_level", {})
        mean_p = float(m.get("mean_paraphrase_similarity", 0.16))
        mean_u = float(m.get("mean_unrelated_similarity", 0.01))
        # Midpoint + margin above unrelated pairs, anchored to paraphrase scale
        derived = round(max(0.18, mean_p + 0.04), 4)
        rationale = (
            f"Derived from Part B module data ({embedding_version}): "
            f"mean_paraphrase={mean_p:.4f}, mean_unrelated={mean_u:.4f} -> threshold={derived:.4f}"
        )
        return derived, rationale
    return 0.20, "Default threshold=0.20 (Part B baseline.json not found)"


def agglomerative_cluster_queries(
    uncovered: list[dict[str, Any]],
    embedding_version: str,
    n_clusters: int = 5,
) -> list[dict[str, Any]]:
    """Cluster all uncovered queries using average-linkage agglomerative clustering on embeddings."""
    if not uncovered:
        return []
    k = min(n_clusters, len(uncovered))
    vecs = [np.asarray(embed_text(u["query"], version=embedding_version), dtype=np.float64) for u in uncovered]
    clusters: list[list[int]] = [[i] for i in range(len(uncovered))]

    while len(clusters) > k:
        best_sim = -2.0
        best_pair = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Average pairwise cosine similarity
                sims = []
                for a in clusters[i]:
                    for b in clusters[j]:
                        na = float(np.linalg.norm(vecs[a]))
                        nb = float(np.linalg.norm(vecs[b]))
                        sims.append(float(np.dot(vecs[a], vecs[b]) / (na * nb)) if na > 0 and nb > 0 else 0.0)
                avg_sim = sum(sims) / len(sims)
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_pair = (i, j)
        i, j = best_pair
        merged = clusters[i] + clusters[j]
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        clusters.append(merged)

    labelled_clusters: list[dict[str, Any]] = []
    for c_idx, member_indices in enumerate(clusters, start=1):
        items = [uncovered[idx] for idx in member_indices]
        term_counts: Counter[str] = Counter()
        for it in items:
            words = re.findall(r"[a-z0-9]{3,}", it["query"].lower())
            for w in words:
                if w not in _STOPWORDS:
                    term_counts[w] += 1
        top_terms = [w for w, _ in term_counts.most_common(4)]
        label = ", ".join(top_terms) if top_terms else f"cluster_{c_idx}"
        labelled_clusters.append(
            {
                "cluster_id": c_idx,
                "label_terms": top_terms,
                "cluster_label": label,
                "size": len(items),
                "queries": items,
            }
        )
    labelled_clusters.sort(key=lambda x: x["size"], reverse=True)
    return labelled_clusters


def _dist_summary(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "min": 0.0, "p50": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "count": len(s),
        "min": round(s[0], 4),
        "p50": round(s[len(s) // 2], 4),
        "max": round(s[-1], 4),
    }


def evaluate_coverage(
    live_queries: list[dict[str, Any]],
    golden_questions: list[dict[str, Any]],
    threshold: float,
    embedding_version: str,
) -> dict[str, Any]:
    golden_embs = [(g["id"], g["question"], embed_text(g["question"], version=embedding_version)) for g in golden_questions]

    covered_count = 0
    uncovered: list[dict[str, Any]] = []
    domain_scores: dict[str, list[float]] = {}

    for uq in live_queries:
        u_text = uq["query"]
        dom = str(uq.get("domain", "unknown"))
        u_emb = embed_text(u_text, version=embedding_version)

        best_score = -1.0
        best_g_id = None
        best_g_q = None
        for g_id, g_q, g_emb in golden_embs:
            sim = cosine_similarity(u_emb, g_emb)
            if sim > best_score:
                best_score = sim
                best_g_id = g_id
                best_g_q = g_q

        domain_scores.setdefault(dom, []).append(best_score)
        if best_score >= threshold:
            covered_count += 1
        else:
            uncovered.append(
                {
                    "id": uq["id"],
                    "domain": dom,
                    "query": u_text,
                    "best_score": round(best_score, 4),
                    "nearest_golden_id": best_g_id,
                    "nearest_golden_q": best_g_q,
                }
            )

    coverage_share = covered_count / len(live_queries) if live_queries else 0.0
    clusters = agglomerative_cluster_queries(uncovered, embedding_version=embedding_version, n_clusters=5)

    return {
        "total_queries": len(live_queries),
        "domain_breakdown": {k: len(v) for k, v in sorted(domain_scores.items())},
        "domain_score_distributions": {k: _dist_summary(v) for k, v in sorted(domain_scores.items())},
        "covered_count": covered_count,
        "uncovered_count": len(uncovered),
        "coverage_share": round(coverage_share, 4),
        "target_threshold": TARGET_COVERAGE_SHARE,
        "similarity_threshold": threshold,
        "gap_detected": coverage_share < TARGET_COVERAGE_SHARE,
        "uncovered_queries": uncovered,
        "clusters": clusters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Part E: Semantic Query Coverage & Drift Monitor")
    parser.add_argument("--golden-path", default="eval/golden.jsonl")
    parser.add_argument("--live-queries", default="drift/user_queries.jsonl")
    parser.add_argument("--embedding-version", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    emb_ver = args.embedding_version or get_embedding_version()
    if args.threshold is not None:
        thresh = args.threshold
        rationale = f"CLI override threshold={thresh:.4f}"
    else:
        thresh, rationale = derive_threshold_from_part_b(emb_ver)

    live_qs = load_jsonl(PROJECT_ROOT / args.live_queries if not Path(args.live_queries).is_absolute() else args.live_queries)
    golden_qs = load_jsonl(PROJECT_ROOT / args.golden_path if not Path(args.golden_path).is_absolute() else args.golden_path)

    res = evaluate_coverage(live_qs, golden_qs, threshold=thresh, embedding_version=emb_ver)
    res["label"] = args.label
    res["golden_path"] = args.golden_path
    res["golden_count"] = len(golden_qs)
    res["embedding_version"] = emb_ver
    res["threshold_rationale"] = rationale
    res["timestamp"] = datetime.datetime.now(timezone.utc).isoformat()

    out_dir = get_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"coverage_{args.label}_{int(time.time())}.json"
    tmp = out_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    os.replace(tmp, out_file)

    dom_str = ", ".join(f"{k}={v}" for k, v in res["domain_breakdown"].items())
    print("\n" + "=" * 82)
    print(f"  RAGOPS LIVE QUERY COVERAGE & DRIFT REPORT [{args.label}]")
    print("=" * 82)
    print(f"Golden Set:         {args.golden_path} ({len(golden_qs)} questions)")
    print(f"Live Queries:       {res['total_queries']} ({dom_str})")
    print(f"Embedding Version:  {emb_ver}")
    print(f"Threshold:          {thresh:.4f} ({rationale})")
    print(f"Score Distributions:")
    for dom, stats in res["domain_score_distributions"].items():
        print(f"  - {dom:<14}: n={stats['count']}, min={stats['min']}, p50={stats['p50']}, max={stats['max']}")
    print(f"Covered Queries:    {res['covered_count']} / {res['total_queries']} ({res['coverage_share'] * 100:.1f}%)")
    print(f"Target Share:       {TARGET_COVERAGE_SHARE * 100:.1f}%")

    if res["uncovered_queries"]:
        print(f"\nUncovered Queries ({res['uncovered_count']} total, all listed):")
        for u in res["uncovered_queries"]:
            print(
                f"  - {u['id']} [{u['domain']}] score={u['best_score']:.4f} "
                f"(nearest={u['nearest_golden_id']}): {u['query']}"
            )

        print(f"\nAgglomerative Clusters ({len(res['clusters'])} clusters, 0 unclustered):")
        for cl in res["clusters"]:
            print(f"  Cluster {cl['cluster_id']} [{cl['cluster_label']}] (size={cl['size']}):")
            for q in cl["queries"]:
                print(f"    * {q['id']}: {q['query']}")

    print(f"\nCoverage report saved to: {out_file}")
    print("=" * 82)

    if res["gap_detected"]:
        print(f"GAP DETECTED ({res['coverage_share'] * 100:.1f}% < 85.0%). Exiting with code 1.")
        sys.exit(1)
    else:
        print(f"COVERAGE TARGET MET ({res['coverage_share'] * 100:.1f}% >= 85.0%). Exiting with code 0.")
        sys.exit(0)


if __name__ == "__main__":
    main()
