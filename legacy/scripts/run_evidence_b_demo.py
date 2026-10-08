#!/usr/bin/env python
"""End-to-end N01 -> N03/N05 -> N08 -> N07 -> N12/N19 (rev 3): image -> X -> concept scores ->
grouped, temporal/polarity-resolved findings (Evidence B). Needs a completed tag sidecar --
run scripts/build_concept_tags.py first (full run: ~2h on CPU at the measured ~50/s throughput;
use --limit for a quick smoke test).

  python scripts/run_evidence_b_demo.py
  python scripts/run_evidence_b_demo.py --tags-path model_weights/concept_tags_smoke.jsonl.gz --dicom-id <id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.concepts.bank import ConceptBank, load_tags  # noqa: E402
from rrg.concepts.grouping import group_and_rank  # noqa: E402
from rrg.concepts.similarity import score_concepts  # noqa: E402
from rrg.core.config import load_settings, require_file  # noqa: E402
from rrg.core.runtime import configure_runtime  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dicom-id", default=None)
    ap.add_argument("--tags-path", default=None, help="override paths.model_weights_dir/concept_tags.jsonl.gz")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    device = configure_runtime(s)
    tags_path = Path(args.tags_path) if args.tags_path else s.paths.model_weights_dir / "concept_tags.jsonl.gz"
    require_file(tags_path, "--tags-path")

    index = MimicCxrIndex.from_settings(s)
    rec = index.get(args.dicom_id) if args.dicom_id else index.select(split=s.demo.split, view_positions=s.dataset.accepted_view_positions)[0]
    dec, meta = load_record_image(rec, s)

    encoder = ClearEncoder(s, device)
    bank = ConceptBank.load(s).with_tags(load_tags(tags_path, s.concept_bank.expected_n_concepts))
    coverage = sum(1 for t in bank.tags if t.group_key is not None) / len(bank.tags)
    print(f"study {meta.study_uid} ({meta.view_position}) | tag sidecar: {len(bank.tags):,} concepts, "
          f"{coverage:.1%} have a group_key")

    emb = encoder.encode(dec)
    scores = score_concepts(emb, bank, device)
    groups = group_and_rank(scores, bank, s.grouping)

    print(f"\nEvidence B: top {len(groups)} finding groups\n" + "=" * 100)
    for g in groups:
        def fmt(score, count):
            return "—" if score != score else f"{score:.3f} (n={count})"   # NaN check
        print(f"[{g.finding}]  anatomy={g.anatomy or '—'}  laterality={g.laterality.value}")
        print(f"    present={fmt(g.present_score, g.present_count)}   absent={fmt(g.absent_score, g.absent_count)}")
        for c in g.top_contributing_concepts:
            print(f"      {c.raw_score:.3f} [{c.resolved_polarity.value:8s} {c.temporal_class:16s}] {c.text}")


if __name__ == "__main__":
    main()
