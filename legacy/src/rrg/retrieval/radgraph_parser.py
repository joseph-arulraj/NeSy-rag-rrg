"""Shared RadGraph wrapper. Used offline (concept-bank tagging, pipeline.md 4.6; reference-
corpus precompute for N15) and online at N29 (draft claim extraction). One wrapper so every
consumer sees the same entity/relation schema.

Uses the `radgraph` pip package (Stanford-AIMI), model_type "modern-radgraph-xl". Verified in
this project (pipeline.md 10.2/GAP-21): runs on CPU (no GPU required), weights download
automatically and openly from Hugging Face on first use (no PhysioNet credential needed for
inference -- that gate applies to the training DATASET, not the released model). Known
behaviour, confirmed empirically, not assumed:
  - Entities: "Anatomy" / "Observation", each with a certainty suffix
    ("definitely present" / "definitely absent" / "uncertain").
  - Relations seen in practice: "located_at", "modify". (pipeline.md's original spec also
    listed "suggestive_of" per the general RadGraph schema; not observed in this project's
    probe and not relied upon here.)
  - Negation is correctly captured in the certainty label. Temporal/comparison language
    ("resolved", "unchanged") is NOT -- confirmed by direct test, see concepts/tagging/polarity.py.
  - Laterality words are never extracted as entities at all.
  - Coverage on short, concept-bank-style noun phrases is materially incomplete: 5 of 13
    hand-picked phrases (including bare "cardiomegaly") returned zero entities in this
    project's probe. Treat every RadGraphParse as possibly empty, not as a reliable oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..core.errors import PipelineError


class RadGraphLoadError(PipelineError):
    code = "E_RADGRAPH_LOAD"


@dataclass(frozen=True)
class RadGraphEntity:
    index: str                   # this entity's key in the raw RadGraph output -- relations refer to it
    text: str
    entity_type: str            # "Observation" | "Anatomy"
    certainty: str               # "definitely present" | "definitely absent" | "uncertain"
    char_span: Optional[tuple[int, int]] = None   # populated when parsing a draft (N29); not meaningful for a bare concept string


@dataclass(frozen=True)
class RadGraphRelation:
    kind: str                    # "located_at" | "modify" (see module docstring)
    source_index: str
    target_index: str


@dataclass(frozen=True)
class RadGraphParse:
    text: str
    entities: list[RadGraphEntity]
    relations: list[RadGraphRelation]

    def observations(self) -> list[RadGraphEntity]:
        return [e for e in self.entities if e.entity_type == "Observation"]

    def anatomy(self) -> list[RadGraphEntity]:
        return [e for e in self.entities if e.entity_type == "Anatomy"]


def _to_parse(text: str, raw: dict) -> RadGraphParse:
    raw_entities: dict[str, dict] = raw.get("entities", {})
    entities: list[RadGraphEntity] = []
    relations: list[RadGraphRelation] = []
    for idx, e in raw_entities.items():
        label = e.get("label", "")
        entity_type, _, certainty = label.partition("::")
        entities.append(RadGraphEntity(index=idx, text=e.get("tokens", ""), entity_type=entity_type, certainty=certainty))
        for rel_kind, target_idx in e.get("relations", []) or []:
            relations.append(RadGraphRelation(kind=rel_kind, source_index=idx, target_index=target_idx))
    return RadGraphParse(text=text, entities=entities, relations=relations)


class RadGraphParser:
    """Loaded once; `radgraph.RadGraph` itself is expensive to construct (loads the model).

    `device`/`batch_size` map onto the real `radgraph.RadGraph.__init__` signature (verified
    directly, not assumed: `(self, batch_size=1, cuda=None, model_type=None, temp_dir=None,
    model_cache_dir=None, tokenizer_cache_dir=None, **kwargs)`) -- confirmed as a real gap that
    every prior call site (concept tagging, calibration, the full pipeline) left unset, so
    `radgraph.device`/`radgraph.batch_size` in config were silently ignored and RadGraph always
    ran on whichever device the package itself defaults to. This affects THROUGHPUT only, not
    correctness -- any tags/facts already produced under the unwired version remain valid, just
    possibly slower than necessary.

    `cuda` is NOT a boolean (confirmed by the package's own source after an initial wrong guess
    crashed with `Invalid device string: 'cuda:True'`) -- it's a GPU device INDEX, used internally
    as `torch.device(f"cuda:{cuda}" if cuda != -1 else "cpu")`. So -1 means CPU, and 0 means the
    first GPU; there is no built-in "auto" in the package itself, so "auto" is resolved here by
    checking `torch.cuda.is_available()` ourselves, the same way `core/runtime.resolve_device`
    does for everything else in this project."""

    def __init__(self, model_type: str = "modern-radgraph-xl", device: Optional[str] = None, batch_size: int = 1):
        try:
            from radgraph import RadGraph
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            if missing and missing != "radgraph":
                # radgraph's bundled allennlp shim needs packages (e.g. "requests") it does not
                # declare as its own dependencies, so `pip install radgraph` alone can leave one
                # missing. The resulting ImportError is easy to misread as "radgraph itself is
                # not installed" when it prints last -- name the actual missing package instead.
                raise RadGraphLoadError(
                    f"'radgraph' is installed but failed to import because it needs {missing!r}, "
                    f"which isn't one of its declared dependencies. Install it directly: "
                    f"pip install {missing}"
                ) from exc
            raise RadGraphLoadError(
                "cannot import 'radgraph'. Install it: pip install radgraph"
            ) from exc
        try:
            kwargs: dict[str, Any] = {"model_type": model_type, "batch_size": batch_size}
            if device is not None:
                if device == "cuda":
                    kwargs["cuda"] = 0
                elif device == "auto":
                    import torch
                    kwargs["cuda"] = 0 if torch.cuda.is_available() else -1
                else:
                    # "cpu", or "mps" -- radgraph's own param is a CUDA device index with no MPS
                    # equivalent, so anything that isn't "cuda"/"auto" falls back to CPU (-1).
                    kwargs["cuda"] = -1
            self._model = RadGraph(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- surface any load/download failure with context
            raise RadGraphLoadError(f"RadGraph failed to load (model_type={model_type!r}): {exc}") from exc
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size

    def parse(self, text: str) -> RadGraphParse:
        return self.parse_batch([text])[0]

    def parse_batch(self, texts: list[str]) -> list[RadGraphParse]:
        if not texts:
            return []
        raw = self._model(list(texts))
        # The package returns a dict keyed "0".."n-1" in input order, per this project's probe.
        return [_to_parse(texts[int(k)], v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
