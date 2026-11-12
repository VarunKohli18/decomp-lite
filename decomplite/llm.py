"""Provider dispatch for LLM calls. Returns a normalized dict with text +
latency + token counts, regardless of provider."""
from __future__ import annotations

import os
import time

import requests


def _base_url(cfg: dict, env: str, default: str) -> str:
    """Host root of the server, no trailing slash and no `/v1`."""
    base = (cfg.get("base_url") or os.environ.get(env) or default).rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _chat_completions(url: str, key: str | None, payload: dict, timeout: float) -> dict:
    """POST an OpenAI-style chat completion and normalize the reply."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM call failed [{r.status_code}]: {r.text[:500]}")
    j = r.json()
    usage = j.get("usage") or {}
    return {
        "text": j["choices"][0]["message"].get("content") or "",
        "latency_s": round(time.time() - t0, 2),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _messages(system: str, user: str) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user}]


def _vllm(system: str, user: str, cfg: dict) -> dict:
    """A self-hosted OpenAI-compatible server: vLLM, LM Studio, llama.cpp."""
    payload = {"model": cfg["model"], "stream": False, "messages": _messages(system, user),
               "temperature": cfg.get("temperature", 0.1)}
    # Output cap. Omitted when the model config sets no `max_tokens`, which
    # leaves the server's own default.
    if cfg.get("max_tokens"):
        payload["max_tokens"] = cfg["max_tokens"]
    # Provider-specific request fields, merged verbatim from the model config.
    payload.update(cfg.get("extra_body") or {})
    # Raw vLLM ports take no key, so the header is sent only when one is set.
    return _chat_completions(
        f"{_base_url(cfg, 'VLLM_BASE_URL', 'http://localhost:8000')}/v1/chat/completions",
        os.environ.get(cfg.get("api_key_env") or "VLLM_API_KEY"), payload,
        float(os.environ.get("VLLM_READ_TIMEOUT", "900")))


def _openai(system: str, user: str, cfg: dict) -> dict:
    """OpenAI's API (GPT models)."""
    env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"{env} is not set (put it in .env)")
    payload = {"model": cfg["model"], "messages": _messages(system, user)}
    # Current GPT models take `max_completion_tokens`, and many reject a
    # non-default temperature, so one is sent only when the entry sets it.
    if cfg.get("max_tokens"):
        payload["max_completion_tokens"] = cfg["max_tokens"]
    if cfg.get("temperature") is not None:
        payload["temperature"] = cfg["temperature"]
    payload.update(cfg.get("extra_body") or {})
    return _chat_completions(
        f"{_base_url(cfg, 'OPENAI_BASE_URL', 'https://api.openai.com')}/v1/chat/completions",
        key, payload, float(os.environ.get("OPENAI_READ_TIMEOUT", "900")))


def _anthropic(system: str, user: str, cfg: dict) -> dict:
    """Anthropic's API (Claude models), through the official SDK."""
    import anthropic

    # Credentials come from the environment (ANTHROPIC_API_KEY in .env).
    client = anthropic.Anthropic(**({"base_url": cfg["base_url"]} if cfg.get("base_url") else {}))
    params = {
        "model": cfg["model"],
        "max_tokens": cfg.get("max_tokens") or 16000,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        params["system"] = system
    # Thinking is adaptive on current Claude models; `effort` sets its depth.
    # No temperature: current models reject sampling parameters.
    if cfg.get("effort"):
        params["output_config"] = {"effort": cfg["effort"]}
    # Server-side fallback: if the model declines a request (reverse
    # engineering can trip its cyber safeguards), another Claude model answers
    # in the same call. `fallbacks: default` in the entry turns it on.
    if cfg.get("fallbacks"):
        params.update(betas=["server-side-fallback-2026-07-01"], fallbacks=cfg["fallbacks"])
        create = client.beta.messages.create
    else:
        create = client.messages.create

    t0 = time.time()
    try:
        r = create(**params)
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Claude rate limit: {e.message}") from None
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude call failed [{e.status_code}]: {e.message}") from None
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Claude connection error: {e}") from None
    if r.stop_reason == "refusal":
        category = getattr(getattr(r, "stop_details", None), "category", None)
        raise RuntimeError(f"Claude declined the request (category: {category})")
    if r.stop_reason == "max_tokens":
        raise RuntimeError("Claude hit max_tokens before finishing; raise it in models.yaml")
    return {
        "text": "".join(b.text for b in r.content if b.type == "text"),
        "latency_s": round(time.time() - t0, 2),
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
        "served_by": r.model,
    }


def _ollama(system: str, user: str, cfg: dict) -> dict:
    """Ollama's native /api/chat, which (unlike its /v1) honours `num_ctx`."""
    options = {"temperature": cfg.get("temperature", 0.1),
               "num_ctx": cfg.get("num_ctx", 16384)}
    if cfg.get("max_tokens"):
        options["num_predict"] = cfg["max_tokens"]
    payload = {
        "model": cfg["model"],
        "stream": False,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": user}],
        "options": options,
    }
    t0 = time.time()
    r = requests.post(f"{_base_url(cfg, 'OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
                      json=payload,
                      timeout=float(os.environ.get("OLLAMA_READ_TIMEOUT", "900")))
    if r.status_code >= 400:
        raise RuntimeError(f"Ollama call failed [{r.status_code}]: {r.text[:500]}")
    j = r.json()
    return {
        "text": j["message"]["content"],
        "latency_s": round(time.time() - t0, 2),
        "input_tokens": j.get("prompt_eval_count", 0),
        "output_tokens": j.get("eval_count", 0),
    }


_PROVIDERS = {"vllm": _vllm, "ollama": _ollama, "openai": _openai, "anthropic": _anthropic}


def call(cfg: dict, system: str, user: str) -> dict:
    """Single entry point for every LLM call. `cfg` is one models.yaml entry."""
    provider = cfg.get("provider")
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}. Known: {sorted(_PROVIDERS)}")
    if not cfg.get("model"):
        raise ValueError("model entry has no `model` (the name the server serves it under)")
    return _PROVIDERS[provider](system, user, cfg)
