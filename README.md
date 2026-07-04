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

> Status: **P0 complete** — config-driven RAG target. P1–P5 (eval harness,
> dashboard, CI gate, monitoring, AutoTuner) in progress.

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
