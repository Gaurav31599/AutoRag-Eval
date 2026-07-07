# AutoRAG-Eval

**Builds AI systems with verification built in.**

**🔴 Live dashboard → [autorag-eval.streamlit.app](https://autorag-eval.streamlit.app/)**

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
- **Self-explaining** — failing cases come with the judge's reason, and a
  context-correctness layer flags answers grounded in stale/wrong retrieved
  context even when faithfulness looks high.

> Status: **P3 complete** — RAG target · eval harness · live dashboard · CI
> regression gate · self-explaining diagnostics. P4–P5 (Langfuse monitoring,
> AutoTuner) in progress.

---

## Versions

Each release has its own focused write-up:

| Version | What it adds | Doc |
|---|---|---|
| **v1** | Config-driven RAG target · RAGAS eval harness · live dashboard | [docs/v1.md](docs/v1.md) |
| **v2** | CI regression gate (RAGAS faithfulness) blocking bad PRs | [docs/v2.md](docs/v2.md) |
| **v3** | Self-explaining failure drill-down · cost-per-run · context-correctness layer | [docs/v3.md](docs/v3.md) |

---

## Architecture

```
config/rag.yaml ──► rag/ (ingest · retrieve · generate)
                      │
                      ├─► eval/  RAGAS 4 metrics · LLM-as-judge · cost · cache · diagnostics
                      ├─► tests/ RAGAS-faithfulness pytest gate ──► GitHub Actions
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

# 5. evaluate + view the dashboard
python -m eval.run --tag baseline
streamlit run dashboard/app.py         # local dashboard at :8501
```

`python -m rag ingest` downloads a 30-question slice of HotpotQA, pools the
paragraphs into a corpus, chunks + embeds them into Chroma, and writes the
human-checkable golden set to [`data/golden/golden.jsonl`](data/golden/).

### Windows note
`make` isn't installed by default on Windows. Every target is a thin wrapper
over `python -m rag ...` / `python -m eval.run ...` — use those directly if you
don't have make.
