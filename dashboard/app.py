"""AutoRAG-Eval dashboard — per-run RAGAS scores, trends, cost, diagnostics.

Reads committed results/*.json only (no API key or vector store required), so
it deploys cleanly to Streamlit Community Cloud.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
FAITHFULNESS_GATE = 0.60  # the CI regression threshold (P2)

st.set_page_config(page_title="AutoRAG-Eval", page_icon="🔎", layout="wide")


@st.cache_data
def load_runs() -> list[dict]:
    runs = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs


def metric_bar(agg: dict) -> go.Figure:
    vals = [agg.get(m) or 0 for m in METRICS]
    colors = ["#2563eb" if (agg.get(m) or 0) >= FAITHFULNESS_GATE else "#f59e0b" for m in METRICS]
    fig = go.Figure(go.Bar(x=METRICS, y=vals, marker_color=colors,
                           text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.add_hline(y=FAITHFULNESS_GATE, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"faithfulness gate {FAITHFULNESS_GATE}")
    fig.update_yaxes(range=[0, 1.05], title="score")
    fig.update_layout(height=360, margin=dict(t=30, b=10), showlegend=False)
    return fig


def trend(runs: list[dict]) -> go.Figure:
    fig = go.Figure()
    x = [r["run_id"] for r in runs]
    for m in METRICS:
        fig.add_trace(go.Scatter(x=x, y=[r["aggregate"].get(m) for r in runs],
                                 mode="lines+markers", name=m))
    fig.add_hline(y=FAITHFULNESS_GATE, line_dash="dash", line_color="#ef4444")
    fig.update_yaxes(range=[0, 1.05], title="score")
    fig.update_layout(height=360, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


st.title("🔎 AutoRAG-Eval")
st.caption("RAG evaluation harness — RAGAS four metrics via LLM-as-judge, with a "
           "context-correctness layer and per-run cost.")

runs = load_runs()
if not runs:
    st.warning("No runs yet. Generate one with `python -m eval.run --tag baseline`.")
    st.stop()

labels = [r["run_id"] for r in runs]
choice = st.sidebar.selectbox("Run", labels, index=len(labels) - 1)
run = next(r for r in runs if r["run_id"] == choice)

st.sidebar.subheader("Config snapshot")
st.sidebar.json(run.get("config", {}))

cost = run.get("cost", {})
st.sidebar.subheader("Cost breakdown (USD)")
st.sidebar.json({
    "generator": cost.get("generator_usd"),
    "judge": cost.get("judge_usd"),
    "diagnostics": cost.get("diagnostics_usd"),
    "total": cost.get("total_usd"),
})

# ── headline metrics ─────────────────────────────────────────────────────────
agg = run["aggregate"]
diag = run.get("diagnostics", {})
cols = st.columns(len(METRICS) + 2)
for col, m in zip(cols, METRICS):
    v = agg.get(m)
    col.metric(m.replace("_", " "), f"{v:.3f}" if v is not None else "n/a")
cols[-2].metric("⚠ stale context", diag.get("stale_context", "—"),
                help="Answers faithful to retrieved context that does NOT actually "
                     "contain the correct answer — caught by the context-correctness layer.")
cols[-1].metric("cost (USD)", f"${cost.get('total_usd', 0):.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Metrics — selected run")
    st.plotly_chart(metric_bar(agg), use_container_width=True)
with right:
    st.subheader("Trend across runs")
    st.plotly_chart(trend(runs), use_container_width=True)

# ── per-question table ───────────────────────────────────────────────────────
st.subheader("Per-question scores")
cases = pd.DataFrame(run["cases"])
table = cases.copy()
if "stale_context" in table.columns:
    table["⚠"] = table["stale_context"].map(lambda x: "⚠" if x else "")
    show = ["⚠", "question", *METRICS]
else:
    show = ["question", *METRICS]
st.dataframe(table[show], use_container_width=True, hide_index=True)

# ── self-explaining drill-down ───────────────────────────────────────────────
st.subheader("Case drill-down")
idx = st.selectbox("Question", range(len(cases)),
                   format_func=lambda i: ("⚠ " if cases.iloc[i].get("stale_context") else "")
                   + cases.iloc[i]["question"])
c = cases.iloc[idx]

d1, d2 = st.columns(2)
d1.markdown(f"**Answer**\n\n{c['answer']}")
d2.markdown(f"**Ground truth**\n\n{c['ground_truth']}")

fv = c.get("faithfulness")
st.markdown(
    f"faithfulness **{fv:.2f}**  ·  answer relevancy **{c.get('answer_relevancy') or 0:.2f}**  "
    f"·  context precision **{c.get('context_precision') or 0:.2f}**  "
    f"·  context recall **{c.get('context_recall') or 0:.2f}**"
    if fv is not None else "no metric scores for this case"
)

# (a) self-explaining failure reason
if c.get("fail_reason"):
    st.error(f"**Why faithfulness failed (judge):** {c['fail_reason']}")

# (c) context-correctness layer
if "context_supported" in cases.columns:
    if c.get("stale_context"):
        st.warning(
            "**⚠ Stale / wrong context:** the answer is faithful to the retrieved "
            "passages, but those passages do **not** actually contain the correct "
            f"answer.\n\n**Auditor:** {c.get('context_reason', '')}"
        )
    elif c.get("context_supported"):
        st.success(f"**Context correctness:** retrieved passages support the correct "
                   f"answer. {c.get('context_reason', '')}")
    else:
        st.info(f"**Context correctness:** passages judged insufficient. "
                f"{c.get('context_reason', '')}")

st.markdown("**Retrieved context**")
for i, ctx in enumerate(c["contexts"], 1):
    st.markdown(f"> **[{i}]** {ctx}")
