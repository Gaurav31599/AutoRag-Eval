"""RAGAS four-metric scoring via a configurable LLM-as-judge.

Judge model comes from config (gpt-4o-mini default; Claude Haiku selectable).
Embeddings for answer-relevancy run locally (sentence-transformers) to stay
free and API-light. Judge token usage is captured so cost.py can price it.
"""
from __future__ import annotations

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = [m.name for m in METRICS]


def _judge_llm(cfg: dict):
    j = cfg["judge"]
    if j["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=j["model"], temperature=j["temperature"])
    if j["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=j["model"], temperature=j["temperature"])
    raise ValueError(f"Unknown judge provider: {j['provider']}")


def _embeddings(cfg: dict):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=cfg["embedding"]["model"],
        model_kwargs={"device": cfg["embedding"]["device"]},
    )


def score(records: list[dict], cfg: dict):
    """Return (pandas DataFrame of per-question scores, judge_usage dict)."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(_judge_llm(cfg))
    emb = LangchainEmbeddingsWrapper(_embeddings(cfg))

    judge_usage = {"input": 0, "output": 0}
    try:
        from langchain_community.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            result = evaluate(dataset, metrics=METRICS, llm=llm, embeddings=emb)
        judge_usage = {"input": cb.prompt_tokens, "output": cb.completion_tokens}
    except Exception:
        # Non-OpenAI judge (callback won't capture) — score without token counts.
        result = evaluate(dataset, metrics=METRICS, llm=llm, embeddings=emb)

    return result.to_pandas(), judge_usage
