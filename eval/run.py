"""Orchestrate one eval run → results/<run_id>.json + a printed summary.

  python -m eval.run                 # full golden set
  python -m eval.run --limit 5       # quick/cheap subset
  python -m eval.run --tag baseline  # label the run
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

from rag.config import load_config, resolve
from eval import cost
from eval.harness import run_pipeline
from eval.metrics import METRIC_NAMES, score

RESULTS_DIR = "results"


def _clean(x):
    """RAGAS can emit NaN for a metric on a degenerate case; JSON-safe it."""
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _config_snapshot(cfg: dict) -> dict:
    return {
        "chunk_size": cfg["chunking"]["chunk_size"],
        "chunk_overlap": cfg["chunking"]["chunk_overlap"],
        "embedding_model": cfg["embedding"]["model"],
        "top_k": cfg["retrieval"]["top_k"],
        "reranker": cfg["retrieval"]["reranker"]["enabled"],
        "generator_model": cfg["generator"]["model"],
        "judge_model": cfg["judge"]["model"],
    }


def run(limit: int | None = None, tag: str = "run") -> dict:
    cfg = load_config()

    print(f"[1/2] Running pipeline over golden set (limit={limit or 'all'})...")
    records, gen_usage = run_pipeline(cfg, limit=limit)

    print(f"[2/2] Scoring {len(records)} answers with RAGAS ({cfg['judge']['model']})...")
    df, judge_usage = score(records, cfg)

    aggregate = {name: _clean(float(df[name].mean())) for name in METRIC_NAMES}

    gen_cost = cost.usd(cfg, cfg["generator"]["model"], gen_usage)
    judge_cost = cost.usd(cfg, cfg["judge"]["model"], judge_usage)

    cases = []
    for rec, (_, row) in zip(records, df.iterrows()):
        cases.append(
            {
                "id": rec.get("id"),
                "question": rec["question"],
                "answer": rec["answer"],
                "ground_truth": rec["ground_truth"],
                "contexts": rec["contexts"],
                **{name: _clean(float(row[name])) for name in METRIC_NAMES},
            }
        )

    run_id = f"{tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "n_cases": len(records),
        "config": _config_snapshot(cfg),
        "aggregate": aggregate,
        "cost": {
            "generator_usd": round(gen_cost, 6),
            "judge_usd": round(judge_cost, 6),
            "total_usd": round(gen_cost + judge_cost, 6),
            "generator_tokens": gen_usage,
            "judge_tokens": judge_usage,
        },
        "cases": cases,
    }

    out_dir = resolve(cfg, RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== RUN {run_id} ===")
    for name in METRIC_NAMES:
        val = aggregate[name]
        print(f"  {name:<22} {val:.3f}" if val is not None else f"  {name:<22} n/a")
    print(f"  {'cost (USD)':<22} {result['cost']['total_usd']:.4f}"
          f"  (gen {gen_cost:.4f} + judge {judge_cost:.4f})")
    print(f"\nSaved -> {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()
    run(limit=args.limit, tag=args.tag)
