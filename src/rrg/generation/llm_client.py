"""N27 `LLM report generation` (pipeline.md §6.1) -- thin client for an OpenAI-compatible
chat/completions API. Currently pointed at KCL's hosted endpoint (model "arc:nexus"), entirely
configured from configs/*.yaml -- no URL or model name is hard-coded here, so swapping providers
later is a config change, not a code change.

The API key is NEVER read from a config file, and never committed: `llm.api_key_env` in config
gives only the NAME of an environment variable, which you must export yourself:

    export RRG_LLM_API_KEY=sk-your-key-here
    python scripts/run_llm_smoke_test.py

This module is a transport layer only -- it does not build prompts from the verified belief
graph (that's N27's other half, still unbuilt: prompt construction needs a real BeliefGraph to
serialise, which needs fusion/N25 to exist first). `LLMClient.generate()` takes a system/user
prompt pair and returns a raw response; nothing here is pipeline-specific.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from ..core.config import Settings
from ..core.errors import LLMMalformedOutputError, LLMUnavailableError


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_hash: str        # pipeline.md §6.1: DraftReport must record this per attempt
    raw: dict[str, Any]     # full provider response, kept for the audit trail


def _read_api_key(api_key_env: str) -> str:
    key = os.environ.get(api_key_env)
    if not key:
        raise LLMUnavailableError(
            f"environment variable {api_key_env!r} (llm.api_key_env) is not set. "
            f"Export it before running: `export {api_key_env}=sk-...` -- the key itself must "
            f"never live in a config file or be committed."
        )
    return key


class LLMClient:
    """Construct once (fails fast if the API key isn't set), call .generate() per report.
    Holds no state between calls beyond the resolved API key -- consistent with pipeline.md
    INV-4 (only the LLM steps may be stochastic; nothing else about the client should
    introduce hidden state across studies)."""

    def __init__(self, settings: Settings):
        self.cfg = settings.llm
        self._api_key = _read_api_key(self.cfg.api_key_env)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Failure paths per pipeline.md §6.1: API error/timeout -> bounded retry with backoff,
        exhausted -> LLMUnavailableError (E_LLM_UNAVAILABLE); empty/malformed response ->
        LLMMalformedOutputError (E_LLM_MALFORMED), NOT retried -- v1 has no regeneration loop
        to recover into (rev 1), so a malformed response fails the study rather than looping."""
        cfg = self.cfg
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        }
        prompt_hash = hashlib.sha256(f"{system_prompt}\x00{user_prompt}".encode("utf-8")).hexdigest()[:16]

        last_exc: Optional[Exception] = None
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = requests.post(
                    cfg.base_url,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=cfg.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < cfg.max_retries:
                    time.sleep(min(2 ** attempt, 30))  # bounded exponential backoff
                    continue
                raise LLMUnavailableError(
                    f"LLM API unreachable after {cfg.max_retries + 1} attempt(s) against {cfg.base_url!r}: {exc}"
                ) from exc
            else:
                return _parse_response(data, cfg.model, prompt_hash)
        raise LLMUnavailableError(f"LLM API unreachable after {cfg.max_retries + 1} attempt(s): {last_exc}") from last_exc


def _parse_response(data: dict[str, Any], model: str, prompt_hash: str) -> LLMResponse:
    try:
        choices = data["choices"]
        if not choices:
            raise LLMMalformedOutputError(f"LLM response had an empty 'choices' list: {data!r}")
        text = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMMalformedOutputError(
            f"LLM response did not match the expected chat/completions shape: {data!r}"
        ) from exc
    if not text or not text.strip():
        raise LLMMalformedOutputError(f"LLM returned an empty message: {data!r}")
    return LLMResponse(text=text, model=model, prompt_hash=prompt_hash, raw=data)
