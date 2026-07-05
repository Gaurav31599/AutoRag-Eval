"""Run the RAG pipeline over the golden set, with answer caching."""
from __future__ import annotations

import json

from rag.config import resolve
from eval import cache

GOLDEN_PATH = "data/golden/golden.jsonl"


def load_golden(cfg: dict, limit: int | None = None) -> list[dict]:
    path = resolve(cfg, GOLDEN_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m rag ingest` first."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def run_pipeline(cfg: dict, limit: int | None = None) -> tuple[list[dict], dict]:
    """Return (records, fresh_generator_usage).

    fresh_generator_usage counts only answers actually generated this run
    (cache hits cost nothing), so per-run $ reflects real spend.
    """
    golden = load_golden(cfg, limit)
    store = cache.load(cfg)
    pipe = None
    records: list[dict] = []
    fresh = {"input": 0, "output": 0}

    for row in golden:
        q = row["question"]
        if q in store:
            rec = store[q]
        else:
            if pipe is None:
                from rag.pipeline import RAGPipeline

                pipe = RAGPipeline(cfg)
            out = pipe.answer(q)
            rec = {"answer": out["answer"], "contexts": out["contexts"], "usage": out["usage"]}
            store[q] = rec
            fresh["input"] += out["usage"]["input"]
            fresh["output"] += out["usage"]["output"]
        records.append({**row, **rec})

    cache.save(cfg, store)
    return records, fresh
