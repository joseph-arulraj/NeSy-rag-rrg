#!/usr/bin/env python
"""Inspect concept_tags.jsonl.gz alongside the concept text it tags.

  python scripts/read_concept_tags.py                                  # random sample of 20
  python scripts/read_concept_tags.py --finding pleural_effusion       # every concept mapped to one finding
  python scripts/read_concept_tags.py --finding cardiomegaly --limit 10
  python scripts/read_concept_tags.py --concept-id 3                   # one specific row
  python scripts/read_concept_tags.py --unmapped                       # concepts with no canonical_finding
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.concepts.bank import load_tags  # noqa: E402
from rrg.concepts.vocabulary import clean_text, load_concept_texts  # noqa: E402
from rrg.core.config import load_settings, require_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--finding", default=None, help="only show concepts mapped to this canonical_finding")
    ap.add_argument("--unmapped", action="store_true", help="only show concepts with no canonical_finding")
    ap.add_argument("--concept-id", type=int, default=None, help="show exactly this one concept")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    texts = load_concept_texts(require_file(s.paths.concepts_csv, "paths.concepts_csv"))
    tags = load_tags(s.paths.model_weights_dir / "concept_tags.jsonl.gz", len(texts))

    if args.concept_id is not None:
        rows = [args.concept_id]
    else:
        candidates = range(len(tags))
        if args.finding:
            candidates = [i for i in candidates if tags[i].canonical_finding == args.finding]
        elif args.unmapped:
            candidates = [i for i in candidates if tags[i].canonical_finding is None]
        candidates = list(candidates)
        print(f"{len(candidates):,} matching concepts")
        rows = random.Random(args.seed).sample(candidates, min(args.limit, len(candidates)))

    for i in rows:
        t = tags[i]
        print(f"[{i:>6}] {clean_text(texts[i])}")
        print(f"         anatomy={t.anatomy}  laterality={t.laterality.value}  "
              f"polarity={t.resolved_polarity.value if t.resolved_polarity else None}  "
              f"temporal={t.temporal_class}  finding={t.canonical_finding}")


if __name__ == "__main__":
    main()
