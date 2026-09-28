#!/usr/bin/env python
"""Smoke-test the N27 LLM client against the configured endpoint, independent of everything
else in the pipeline (no belief graph, no evidence -- just: does the API key + endpoint +
model name actually work).

  export RRG_LLM_API_KEY=sk-your-key-here
  python scripts/run_llm_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402
from rrg.generation.llm_client import LLMClient  # noqa: E402


def main() -> None:
    s = load_settings()
    print(f"provider={s.llm.provider} model={s.llm.model} base_url={s.llm.base_url}")
    client = LLMClient(s)
    resp = client.generate(
        system_prompt="You are a helpful research assistant.",
        user_prompt="In one sentence, what is the significance of attention mechanisms in transformers?",
    )
    print(f"prompt_hash={resp.prompt_hash}")
    print(resp.text)


if __name__ == "__main__":
    main()
