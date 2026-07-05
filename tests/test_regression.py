"""RAGAS faithfulness regression gate over the golden set.

Runs the REAL RAG pipeline with the *current* config, scores faithfulness with
RAGAS (LLM-as-judge), and fails if the mean drops below the gate. A PR that
regresses the config (e.g. shrinks chunk_size) makes answers less grounded in
the retrieved context, faithfulness drops, and CI blocks the merge.

Why RAGAS, not DeepEval, for the gate:
  DeepEval faithfulness only penalises claims that *contradict* the context, so
  it stays ~0.75 even when retrieval degrades — blind to a chunk-size
  regression. RAGAS faithfulness penalises *unsupported* claims, so it drops
  (measured baseline 0.69 -> 0.52 at chunk_size=200). See README P2.

Gate is 0.60, not the brief's default 0.80: HotpotQA is multi-hop and a cheap
judge caps faithfulness ~0.7. Tuned baseline sits ~0.69 (passes with margin);
the chunk-200 regression lands ~0.52 (blocked). We score the whole golden set —
an 8-case subset has too much run-to-run variance to gate on.
"""
from __future__ import annotations

import math
import os

import pytest

from eval.harness import run_pipeline
from eval.metrics import score_faithfulness
from rag.config import load_config

FAITHFULNESS_GATE = 0.60

_env = os.environ.get("CI_SUBSET")
CI_SUBSET = int(_env) if _env else None  # None = full golden set


@pytest.fixture(scope="module")
def scored() -> list[tuple[dict, float | None]]:
    cfg = load_config()
    records, _ = run_pipeline(cfg, limit=CI_SUBSET)
    values = score_faithfulness(records, cfg)
    return list(zip(records, values))


def test_faithfulness_gate(scored: list[tuple[dict, float | None]]) -> None:
    valid = [(r, v) for r, v in scored if v is not None and not math.isnan(v)]
    assert valid, "No faithfulness scores were produced."
    mean = sum(v for _, v in valid) / len(valid)

    print("\n=== RAGAS faithfulness ===")
    for r, v in scored:
        if v is None or math.isnan(v):
            print(f"[SKIP] ---   {r['question'][:55]}")
            continue
        flag = "OK " if v >= FAITHFULNESS_GATE else "LOW"
        print(f"[{flag}] {v:.2f}  {r['question'][:55]}")
        if v < FAITHFULNESS_GATE:
            print(f"        answer: {r['answer'][:80]}")
    print(f"\nMEAN faithfulness = {mean:.3f} over {len(valid)} cases  (gate {FAITHFULNESS_GATE})")

    assert mean >= FAITHFULNESS_GATE, (
        f"Faithfulness regressed: mean {mean:.3f} < gate {FAITHFULNESS_GATE}. "
        "A config change made answers less grounded in the retrieved context."
    )
