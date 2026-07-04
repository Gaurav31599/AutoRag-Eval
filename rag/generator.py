"""API-based answer synthesis. Provider + model come from config.

Returns (answer, usage) where usage = {"input": int, "output": int} so the
eval harness (P3) can log tokens + $ per run.
"""
from __future__ import annotations


def generate(cfg: dict, question: str, contexts: list[str]) -> tuple[str, dict]:
    g = cfg["generator"]
    prompt = g["prompt_template"].format(
        context="\n\n".join(contexts), question=question
    )

    if g["provider"] == "openai":
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=g["model"],
            temperature=g["temperature"],
            max_tokens=g["max_tokens"],
            messages=[{"role": "user", "content": prompt}],
        )
        answer = (resp.choices[0].message.content or "").strip()
        usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
        return answer, usage

    if g["provider"] == "anthropic":
        import anthropic

        resp = anthropic.Anthropic().messages.create(
            model=g["model"],
            max_tokens=g["max_tokens"],
            temperature=g["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text.strip()
        usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        return answer, usage

    raise ValueError(f"Unknown generator provider: {g['provider']}")
