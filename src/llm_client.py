"""Thin LLM client supporting OpenAI (direct) and OpenRouter (via OpenAI SDK).

Adds disk caching keyed by (provider, model, messages, params) and exponential
backoff retry. Designed so repeated runs of the experiment are nearly free.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    error: str | None = None


def _cache_key(provider: str, model: str, messages: list[dict],
               temperature: float, max_tokens: int) -> str:
    blob = json.dumps(
        {"p": provider, "m": model, "msg": messages,
         "t": temperature, "k": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> LLMResult | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return LLMResult(
            text=d["text"], model=d["model"], provider=d["provider"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cached=True, error=d.get("error"),
        )
    except Exception:
        return None


def _save_cache(key: str, result: LLMResult) -> None:
    _cache_path(key).write_text(json.dumps({
        "text": result.text,
        "model": result.model,
        "provider": result.provider,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "error": result.error,
    }))


_OPENAI_CLIENT: OpenAI | None = None
_OPENROUTER_CLIENT: OpenAI | None = None


def _get_client(provider: str) -> OpenAI:
    global _OPENAI_CLIENT, _OPENROUTER_CLIENT
    if provider == "openai":
        if _OPENAI_CLIENT is None:
            _OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return _OPENAI_CLIENT
    if provider == "openrouter":
        if _OPENROUTER_CLIENT is None:
            _OPENROUTER_CLIENT = OpenAI(
                api_key=os.environ["OPENROUTER_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
        return _OPENROUTER_CLIENT
    raise ValueError(f"unknown provider: {provider}")


def chat(provider: str, model: str, system: str, user: str,
         temperature: float = 0.0, max_tokens: int = 256,
         retries: int = 4) -> LLMResult:
    """Single chat completion with cache + retry."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    key = _cache_key(provider, model, messages, temperature, max_tokens)
    cached = _load_cache(key)
    if cached is not None:
        return cached

    client = _get_client(provider)
    last_err: str | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = getattr(resp, "usage", None)
            result = LLMResult(
                text=text,
                model=model,
                provider=provider,
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                cached=False,
                error=None,
            )
            _save_cache(key, result)
            return result
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (2 ** attempt))
    # all retries failed — record as error result, do NOT cache
    return LLMResult(text="", model=model, provider=provider,
                     input_tokens=0, output_tokens=0, cached=False,
                     error=last_err)


if __name__ == "__main__":
    r = chat("openai", "gpt-4.1-mini",
             "You are a helpful assistant.",
             "What is 2+2? Reply with just the digit.",
             max_tokens=20)
    print("text:", repr(r.text))
    print("tokens:", r.input_tokens, "->", r.output_tokens)
    print("cached:", r.cached, "error:", r.error)
