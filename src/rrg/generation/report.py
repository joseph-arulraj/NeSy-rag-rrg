"""N27 -> N28 `Draft report` (pipeline.md §6.1). Wires prompt.py's INV-3-compliant prompt
construction into the transport-only LLMClient (llm_client.py)."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.types import BeliefGraph
from .llm_client import LLMClient
from .prompt import SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class DraftReport:
    text: str
    model: str
    prompt_hash: str
    section_text: str   # alias kept explicit for readability at call sites; same as .text in v1 (no section split yet)


def generate_draft_report(graph: BeliefGraph, client: LLMClient) -> DraftReport:
    if graph.graph_version != "verified":
        raise ValueError(
            f"N27 must receive a VERIFIED belief graph (pipeline.md INV-3), got graph_version={graph.graph_version!r}"
        )
    user_prompt = build_user_prompt(graph)
    resp = client.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    return DraftReport(text=resp.text, model=resp.model, prompt_hash=resp.prompt_hash, section_text=resp.text)
