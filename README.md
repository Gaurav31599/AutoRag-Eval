# AutoRAG-Eval

**Builds AI systems with verification built in.**

A RAG evaluation harness that grows into an autonomous config optimizer. It
evaluates a retrieval-augmented pipeline on canonical metrics, gates
regressions in CI, surfaces a live dashboard, and closes into a loop that tunes
the pipeline config on its own.

- **API-first & cost-aware** — the generator and the LLM-as-judge both call an
  API (default `gpt-4o-mini`; Claude Haiku selectable). Runs on an 8 GB laptop
  GPU; no local model required. Tokens and $ are logged per run.
- **Everything is config-driven** — every knob (chunk size, overlap, top-k,
  reranker, prompt, embedding model) lives in one [`config/rag.yaml`](config/rag.yaml).
  The AutoTuner perturbs that file; nothing is hardcoded.

> Status: **P1 complete** — config-driven RAG + RAGAS eval harness + live
> dashboard. P2–P5 (CI gate, differentiators, monitoring, AutoTuner) in progress.

---

## Architecture (target)

```
config/rag.yaml ──► rag/ (ingest · retrieve · generate)
                      │
                      ├─► eval/  RAGAS 4 metrics · LLM-as-judge · cost · cache
                      ├─► tests/ DeepEval pytest gate  ──► GitHub Actions
                      ├─► dashboard/ Streamlit + Plotly ──► Streamlit Cloud
                      └─► tuner/  propose → run → score → keep/discard (P5)
```

Stack: Chroma · sentence-transformers · RAGAS · DeepEval · Streamlit + Plotly ·
GitHub Actions · Langfuse · Docker. Corpus: a subset of **HotpotQA**
(multi-hop, so retrieval quality genuinely moves the metrics).

---

## Quickstart

```bash
# 1. install
python -m pip install -r requirements.txt        # or: make setup

# 2. add your key
cp .env.example .env        # then edit: OPENAI_API_KEY=sk-...

# 3. build the corpus + golden set + vector store
python -m rag ingest        # or: make ingest

# 4. ask a question (retrieve + answer)
python -m rag "who directed the 1966 batman movie"
#   or: make rag q="who directed the 1966 batman movie"
```

`python -m rag ingest` downloads a 30-question slice of HotpotQA, pools the
paragraphs into a corpus, chunks + embeds them into Chroma, and writes the
human-checkable golden set to [`data/golden/golden.jsonl`](data/golden/).

A query prints the generated answer, the retrieved context, and token usage:

```
=== ANSWER ===
Leslie H. Martinson directed the 1966 Batman movie.

=== RETRIEVED CONTEXT ===
[1] Batman (1966 film) is a ... directed by Leslie H. Martinson ...
...

=== TOKENS ===  in=812  out=14
```

### Windows note
`make` isn't installed by default on Windows. Every target is a thin wrapper
over `python -m rag ...` — use those directly if you don't have make.

---

## P0 — Config-driven RAG target ✅

- One YAML holds every knob (`config/rag.yaml`).
- `rag/ingest.py` — HotpotQA slice → paragraph corpus → char-chunk → embed →
  Chroma; writes the golden Q/A set.
- `rag/retriever.py` — query embed → Chroma top-k → optional cross-encoder reranker.
- `rag/generator.py` — API answer synthesis (OpenAI / Anthropic), returns token usage.
- `rag/pipeline.py` — `RAGPipeline(cfg).answer(q)` → answer + contexts + usage.

**DoD:** `make rag "question"` (or `python -m rag "question"`) returns an
answer plus the retrieved context. ✅

---

## P1 — Eval harness + dashboard ✅

RAGAS four metrics via LLM-as-judge over the golden set, plus a live dashboard.

```bash
python -m eval.run --tag baseline      # score the full golden set
python -m eval.run --limit 5           # quick/cheap subset
streamlit run dashboard/app.py         # local dashboard at :8501
```

- `eval/harness.py` — runs the pipeline over the golden set with an **answer
  cache** (hash of the retrieval+generation knobs); re-running the same config
  re-scores for free.
- `eval/metrics.py` — RAGAS `faithfulness`, `answer_relevancy`,
  `context_precision`, `context_recall`. Judge model from config
  (`gpt-4o-mini` default, Claude Haiku selectable); answer-relevancy embeddings
  run locally (sentence-transformers) to stay API-light.
- `eval/cost.py` — per-run **tokens + USD** (generator + judge), logged into
  every result file.
- `dashboard/app.py` — Streamlit + Plotly: per-run metric cards, a bar chart
  against the faithfulness gate, a cross-run trend, and a per-question
  drill-down. Reads only committed `results/*.json`, so it deploys with no API
  key or vector store.

### The harness earning its keep

Tuning the config knobs and re-measuring drove a real, verifiable improvement on
the (hard, multi-hop) HotpotQA slice — exactly the loop the AutoTuner automates
in P5:

| Config | faithfulness | answer relevancy | context recall |
|---|---|---|---|
| chunk 512 · top-k 4 · strict prompt | 0.36 | 0.18 | 0.57 |
| chunk 1200 · top-k 8 | 0.49 | 0.26 | 0.73 |
| **+ multi-hop synthesis prompt · reranker on** | **0.69** | **0.60** | **0.77** |

A full 30-question run costs **~$0.06** on the default `gpt-4o-mini` judge.

**DoD:** dashboard live at a public URL, reproducible on a fresh checkout. ✅
(deploy steps below)

---

## Deploy the dashboard (Streamlit Community Cloud)

The dashboard reads committed `results/*.json` only — no secrets, no vector
store — so deployment is a one-time click-through:

1. Push this repo to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → pick the repo, branch `main`, main file path
   `dashboard/app.py`.
4. **Deploy.** No secrets needed. You get a public `*.streamlit.app` URL.

To refresh the deployed data, run `python -m eval.run --tag baseline` locally
and push the new `results/baseline-*.json`.
