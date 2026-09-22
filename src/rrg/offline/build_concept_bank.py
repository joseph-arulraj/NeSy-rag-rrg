"""N08/N12/N19 offline tagging pipeline (pipeline.md 4.6 rev 3, steps 1-8): parse all 368,294
concepts with RadGraph, tag anatomy/laterality/polarity/temporal_class/canonical_finding, and
write the per-concept sidecar consumed by concepts/bank.py at runtime.

Resumable: partial results are checkpointed to disk every `tagging.checkpoint_every_batches`
RadGraph batches, so an interrupted HPC job restarts from where it left off rather than
re-parsing 368k concepts from scratch.

  python scripts/build_concept_tags.py --config configs/hpc.yaml
  python scripts/build_concept_tags.py --limit 500          # smoke test
"""
from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from ..concepts.bank import ConceptTag, group_key_for, save_tags
from ..concepts.tagging.finding_vocab import FindingVocabulary, load_finding_vocabulary, pick_core_observation
from ..concepts.tagging.laterality import tag_laterality
from ..concepts.tagging.polarity import classify_temporal, resolve_base_polarity, resolve_polarity
from ..concepts.vocabulary import load_concept_texts
from ..core.config import Settings, require_file
from ..core.types import Laterality, Polarity
from ..fusion.rules.radlex_client import RadLexClient
from ..retrieval.radgraph_parser import RadGraphParser

_CHUNK_PREFIX = "chunk_"


@dataclass(frozen=True)
class TaggingCoverageReport:
    n_concepts: int
    n_with_anatomy: int
    n_with_canonical_finding: int
    n_with_resolved_polarity: int    # excludes concepts dropped for indeterminate/ambiguous temporal language
    n_radgraph_zero_entities: int    # RadGraph returned no entities at all for this concept (GAP-21)
    temporal_class_counts: dict[str, int]
    laterality_counts: dict[str, int]


def _chunk_path(checkpoint_dir: Path, start: int) -> Path:
    return checkpoint_dir / f"{_CHUNK_PREFIX}{start:07d}.jsonl.gz"


def _write_chunk(path: Path, tags: list[ConceptTag]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for t in tags:
            fh.write(json.dumps({
                "anatomy": t.anatomy, "laterality": t.laterality.value,
                "resolved_polarity": t.resolved_polarity.value if t.resolved_polarity else None,
                "temporal_class": t.temporal_class, "canonical_finding": t.canonical_finding,
                "group_key": t.group_key,
            }, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _read_chunk(path: Path) -> list[ConceptTag]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in fh:
            d = json.loads(row)
            out.append(ConceptTag(
                anatomy=d["anatomy"], laterality=Laterality(d["laterality"]),
                resolved_polarity=None if d["resolved_polarity"] is None else Polarity(d["resolved_polarity"]),
                temporal_class=d["temporal_class"], canonical_finding=d["canonical_finding"], group_key=d["group_key"],
            ))
    return out


def _existing_chunks(checkpoint_dir: Path) -> dict[int, Path]:
    if not checkpoint_dir.is_dir():
        return {}
    out = {}
    for p in checkpoint_dir.glob(f"{_CHUNK_PREFIX}*.jsonl.gz"):
        try:
            start = int(p.stem.split(".")[0][len(_CHUNK_PREFIX):])
        except ValueError:
            continue
        out[start] = p
    return out


def _resolve_anatomy(parse, radlex: RadLexClient) -> Optional[str]:
    """First chest-scope-resolvable Anatomy entity, longest text first (a multi-word entity is
    more specific than a single word)."""
    for e in sorted(parse.anatomy(), key=lambda a: -len(a.text)):
        rid = radlex.resolve_anatomy_term(e.text)
        if rid is not None:
            return rid
    return None


def tag_one(
    text: str,
    parse,   # RadGraphParse | None
    radlex: RadLexClient,
    vocab: FindingVocabulary,
    cfg: Settings,
) -> ConceptTag:
    tg = cfg.tagging
    laterality = tag_laterality(text)

    anatomy = _resolve_anatomy(parse, radlex) if parse is not None else None

    observations = parse.observations() if parse is not None else []
    core = pick_core_observation(observations, parse.relations) if parse is not None else None
    certainty = core.certainty if core is not None else None
    base_polarity = resolve_base_polarity(text, tg.negation_cues, certainty)
    temporal = classify_temporal(text, tg.temporal_implies_present, tg.temporal_implies_absent, tg.temporal_indeterminate)
    resolved_polarity = resolve_polarity(base_polarity, temporal)

    canonical_finding = vocab.resolve_observation(core.text) if core is not None else None
    if canonical_finding is None:
        # fallback: try the whole concept text against the synonym table directly (catches
        # e.g. "cardiomegaly" when RadGraph produced no Observation entity at all, GAP-21)
        canonical_finding = vocab.resolve_observation(text)

    return ConceptTag(
        anatomy=anatomy, laterality=laterality, resolved_polarity=resolved_polarity,
        temporal_class=temporal, canonical_finding=canonical_finding,
        group_key=group_key_for(canonical_finding, anatomy, laterality),
    )


def tag_concept_bank(
    texts: list[str],
    radgraph: RadGraphParser,
    radlex: RadLexClient,
    vocab: FindingVocabulary,
    cfg: Settings,
    checkpoint_dir: Path,
    log: Callable[[str], None] = print,
) -> tuple[list[ConceptTag], TaggingCoverageReport]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_chunks(checkpoint_dir)
    tg = cfg.tagging
    bs = cfg.radgraph.batch_size
    n = len(texts)
    n_zero_entities = 0
    t0 = time.time()
    done_batches_since_checkpoint = 0

    all_tags: list[Optional[ConceptTag]] = [None] * n
    for start in range(0, n, bs):
        end = min(start + bs, n)
        chunk_path = _chunk_path(checkpoint_dir, start)
        if start in existing:
            chunk_tags = _read_chunk(chunk_path)
            if len(chunk_tags) != end - start:
                raise ValueError(f"checkpoint {chunk_path} has {len(chunk_tags)} rows, expected {end - start}; delete it and rerun")
        else:
            batch_texts = texts[start:end]
            parses = radgraph.parse_batch(batch_texts)
            n_zero_entities += sum(1 for p in parses if not p.entities)
            chunk_tags = [tag_one(t, p, radlex, vocab, cfg) for t, p in zip(batch_texts, parses)]
            _write_chunk(chunk_path, chunk_tags)
            done_batches_since_checkpoint += 1
        all_tags[start:end] = chunk_tags

        if (start // bs + 1) % max(1, tg.checkpoint_every_batches // 10) == 0 or end == n:
            done = end
            rate = done / max(time.time() - t0, 1e-9)
            log(f"  tagged {done:,}/{n:,}  ({rate:,.1f}/s, eta {(n - done) / max(rate, 1e-9) / 60:.1f} min)")

    tags: list[ConceptTag] = all_tags  # type: ignore[assignment]
    report = TaggingCoverageReport(
        n_concepts=n,
        n_with_anatomy=sum(1 for t in tags if t.anatomy is not None),
        n_with_canonical_finding=sum(1 for t in tags if t.canonical_finding is not None),
        n_with_resolved_polarity=sum(1 for t in tags if t.resolved_polarity is not None),
        n_radgraph_zero_entities=n_zero_entities,
        temporal_class_counts=_count(t.temporal_class for t in tags),
        laterality_counts=_count(t.laterality.value for t in tags),
    )
    return tags, report


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def main(settings: Settings, limit: Optional[int] = None, log: Callable[[str], None] = print) -> TaggingCoverageReport:
    csv_path = require_file(settings.paths.concepts_csv, "paths.concepts_csv")
    texts = list(load_concept_texts(csv_path, settings.concept_bank.expected_n_concepts if limit is None else None))
    if limit is not None:
        texts = texts[:limit]
    log(f"tagging {len(texts):,} concepts")

    radlex = RadLexClient.load(
        settings.radlex.snapshot_path, settings.radlex.chest_scope_root_label,
        settings.radlex.version, tuple(settings.radlex.additional_scope_roots),
    )
    vocab = load_finding_vocabulary(settings.tagging.finding_synonyms_path)
    radgraph = RadGraphParser(model_type=settings.radgraph.model_type)

    checkpoint_dir = settings.tagging.checkpoint_dir / ("smoke" if limit is not None else "full")
    tags, report = tag_concept_bank(texts, radgraph, radlex, vocab, settings, checkpoint_dir, log=log)

    out_path = settings.paths.model_weights_dir / ("concept_tags_smoke.jsonl.gz" if limit is not None else "concept_tags.jsonl.gz")
    save_tags(out_path, tags)
    log(f"wrote {out_path}")
    log(json.dumps(asdict(report), indent=2))
    return report


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rrg.core.config import load_settings  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(load_settings(args.config, args.overrides), args.limit)
