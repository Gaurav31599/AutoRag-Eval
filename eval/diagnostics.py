"""P3 differentiators — diagnostic layers on top of the RAGAS metrics.

Two independent judge calls (kept separate from faithfulness on purpose):

- context_correctness: audits whether the *retrieved passages* actually contain
  the information needed to reach the known-correct answer. This is the "second
  layer" — a RAG can be perfectly faithful to context that is stale or wrong,
  and faithfulness alone will not catch it. When context is unsupported yet
  faithfulness is high, we raise a `stale_context` flag.

- explain_failure: a one-sentence, self-explaining reason a low-faithfulness
  answer failed — surfaced per case in the dashboard drill-down.

Both use the configured judge (gpt-4o-mini default). Calls are cheap and only
the failure explanation is gated to failing cases, to stay cost-aware.
"""
from __future__ import annotations

import json

CTX_SYSTEM = (
    "You are a retrieval auditor for a RAG system. Given a question, the "
    "known-correct answer, and the passages the system retrieved, decide "
    "whether those passages actually CONTAIN the information needed to reach "
    "the correct answer. Judge only the retrieved context, not any generated "
    "answer. Be strict: partial or tangential coverage counts as unsupported."
)

FAIL_SYSTEM = (
    "You are an evaluation analyst. An automated faithfulness grader scored this "
    "answer LOW. In ONE sentence, identify the specific claim or detail a strict "
    "grader would flag as unsupported or only partially supported by the "
    "retrieved passages — even if the answer is arguably correct. Do not say the "
    "answer is fully faithful; name the weakest-grounded part."
)


def _judge_json(cfg: dict, system: str, user: str) -> tuple[dict, dict]:
    j = cfg["judge"]
    if j["provider"] == "openai":
        from openai import OpenAI

        r = OpenAI().chat.completions.create(
            model=j["model"],
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        data = json.loads(r.choices[0].message.content or "{}")
        usage = {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}
        return data, usage

    if j["provider"] == "anthropic":
        import anthropic

        r = anthropic.Anthropic().messages.create(
            model=j["model"],
            max_tokens=300,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user + "\n\nRespond with ONLY a JSON object."}],
        )
        data = json.loads(r.content[0].text)
        usage = {"input": r.usage.input_tokens, "output": r.usage.output_tokens}
        return data, usage

    raise ValueError(f"Unknown judge provider: {j['provider']}")


def context_correctness(cfg: dict, rec: dict) -> tuple[bool, str, dict]:
    """Return (supported, reason, judge_usage)."""
    passages = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(rec["contexts"], 1))
    user = (
        f"Question: {rec['question']}\n"
        f"Known-correct answer: {rec['ground_truth']}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        'Return JSON: {"supported": true|false, "reason": "<one sentence>"}'
    )
    data, usage = _judge_json(cfg, CTX_SYSTEM, user)
    return bool(data.get("supported", False)), str(data.get("reason", "")), usage


def explain_failure(cfg: dict, rec: dict) -> tuple[str, dict]:
    """Return (one_sentence_reason, judge_usage) for a low-faithfulness case."""
    passages = "\n\n".join(rec["contexts"])
    user = (
        f"Question: {rec['question']}\n"
        f"Answer given: {rec['answer']}\n\n"
        f"Retrieved context:\n{passages}\n\n"
        'Return JSON: {"reason": "<one sentence>"}'
    )
    data, usage = _judge_json(cfg, FAIL_SYSTEM, user)
    return str(data.get("reason", "")), usage
