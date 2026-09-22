#!/usr/bin/env python
"""Run N01 -> N03/N05 -> N08/N07 -> N12/N19 (Evidence B) across many images and save both the
raw 368k concept scores (top-N, per image) and the grouped output, so the whole chain from raw
scores through grouping is inspectable at scale, not just for one demo image.

  python scripts/run_evidence_b_batch.py --split train --num-images 10000
  python scripts/run_evidence_b_batch.py --split test --num-images 100 --save-full-scores 3

Output: one JSON Lines file, <output_dir>/evidence_b_batch/<split>.jsonl, one row per study:
  {study_uid, image_id, top_raw_concepts: [{concept_id, text, score}, ...],   # top raw_top_n, before grouping
   evidence_b: [FindingGroup as dict, ...]}                                   # after grouping
Plus, for the first --save-full-scores studies, the complete dense 368,294-d raw score vector
as <output_dir>/evidence_b_batch/raw_scores/<dicom_id>.npy (fp16 -- full fp32 for many studies
is impractical to store, pipeline.md 4.5's storage guidance).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rrg.concepts.bank import ConceptBank, load_tags  # noqa: E402
from rrg.concepts.grouping import group_and_rank  # noqa: E402
from rrg.concepts.similarity import score_concepts_batch, top_k  # noqa: E402
from rrg.concepts.vocabulary import clean_text  # noqa: E402
from rrg.core.config import load_settings, require_file  # noqa: E402
from rrg.core.runtime import configure_runtime  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-images", type=int, default=10000)
    ap.add_argument("--raw-top-n", type=int, default=50, help="how many raw (pre-grouping) top concepts to save per image")
    ap.add_argument("--save-full-scores", type=int, default=0, help="also save the full 368,294-d raw vector for the first N images")
    ap.add_argument("--tags-path", default=None)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    device = configure_runtime(s)
    tags_path = Path(args.tags_path) if args.tags_path else s.paths.model_weights_dir / "concept_tags.jsonl.gz"
    require_file(tags_path, "--tags-path")

    records, _ = MimicCxrIndex.from_settings(s).study_records(args.split)
    records = records[: args.num_images]
    if not records:
        sys.exit(f"no images in split {args.split!r}")

    encoder = ClearEncoder(s, device)
    bank = ConceptBank.load(s).with_tags(load_tags(tags_path, s.concept_bank.expected_n_concepts))
    index = bank.grouping_index()   # built ONCE, reused for every image below
    n_tagged = sum(1 for t in bank.tags if t.group_key is not None)
    print(f"bank: {len(bank):,} concepts, {n_tagged:,} ({n_tagged/len(bank):.1%}) have a group_key, "
          f"{index.n_groups:,} distinct groups | {len(records):,} images from split={args.split!r}")

    out_dir = s.paths.output_dir / "evidence_b_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_scores_dir = out_dir / "raw_scores"
    if args.save_full_scores:
        full_scores_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.jsonl"

    t0 = time.time()
    n_saved_full = 0
    with out_path.open("w", encoding="utf-8") as fh:
        bs = s.clear.batch_size
        for start in range(0, len(records), bs):
            batch = records[start : start + bs]
            decoded = [load_record_image(r, s)[0] for r in batch]
            feats = encoder.encode_batch(decoded)
            scores_batch = score_concepts_batch(feats, bank, device, s.concept_bank.score_batch_size)

            for rec, scores in zip(batch, scores_batch):
                raw_top = top_k(scores, bank, args.raw_top_n)
                groups = group_and_rank(scores, bank, s.grouping, index=index)
                row = {
                    "study_uid": rec.study_id, "image_id": rec.dicom_id, "subject_id": rec.subject_id,
                    "top_raw_concepts": [{"concept_id": h.concept_id, "text": clean_text(h.text), "score": h.score} for h in raw_top],
                    "evidence_b": [
                        {**{k: v for k, v in asdict(g).items() if k != "laterality"}, "laterality": g.laterality.value,
                         "top_contributing_concepts": [
                             {**c, "resolved_polarity": c["resolved_polarity"].value if hasattr(c["resolved_polarity"], "value") else c["resolved_polarity"]}
                             for c in (asdict(cc) for cc in g.top_contributing_concepts)
                         ]}
                        for g in groups
                    ],
                }
                fh.write(json.dumps(row) + "\n")
                if n_saved_full < args.save_full_scores:
                    np.save(full_scores_dir / f"{rec.dicom_id}.npy", scores.astype(np.float16))
                    n_saved_full += 1

            done = start + len(batch)
            if done % args.log_every < bs or done == len(records):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done:,}/{len(records):,}  ({rate:.1f} img/s, eta {(len(records)-done)/max(rate,1e-9)/60:.1f} min)")

    print(f"wrote {out_path}" + (f" and {n_saved_full} full raw-score .npy files under {full_scores_dir}" if n_saved_full else ""))


if __name__ == "__main__":
    main()
