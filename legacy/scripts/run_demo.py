#!/usr/bin/env python
"""End-to-end check of N01 -> N03/N05 -> N08 -> N07 on MIMIC-CXR images.

  python scripts/run_demo.py                                 # Mac: MPS on the 100-study sample
  python scripts/run_demo.py --config configs/hpc.yaml       # H100 profile
  python scripts/run_demo.py --dicom-id <id> --top-k 30      # one specific image
  python scripts/run_demo.py --set demo.num_images=10 --set demo.split=test

For each image it prints the metadata, the CheXpert labels of its study (when available) and the
top-K concepts by cosine similarity, and saves the results to <output_dir>/demo_top_concepts.json.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.concepts.bank import ConceptBank  # noqa: E402
from rrg.concepts.similarity import score_concepts_batch, top_k  # noqa: E402
from rrg.concepts.vocabulary import clean_text  # noqa: E402
from rrg.core.config import load_settings  # noqa: E402
from rrg.core.runtime import configure_runtime  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402


def chexpert_positive_labels(path: Path) -> dict[tuple[int, int], list[str]]:
    """{(subject_id, study_id): [labels marked positive (1.0)]}. Display only; the CBM branch uses these later."""
    out: dict[tuple[int, int], list[str]] = {}
    if not path.is_file():
        return out
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pos = [k for k, v in row.items() if k not in ("subject_id", "study_id") and v.strip() == "1.0"]
            out[(int(row["subject_id"]), int(row["study_id"]))] = pos
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dicom-id", default=None, help="run this one image instead of the first demo.num_images of demo.split")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    device = configure_runtime(s)
    k = args.top_k or s.demo.top_k
    print(f"device={device}  config profile={args.config or 'default'}")

    index = MimicCxrIndex.from_settings(s)
    print("dataset:", index.summary())
    if args.dicom_id:
        records = [index.get(args.dicom_id)]
    else:
        records = index.select(split=s.demo.split, view_positions=s.dataset.accepted_view_positions)[: s.demo.num_images]
    if not records:
        sys.exit(f"no images for split={s.demo.split!r} views={s.dataset.accepted_view_positions}")

    t = time.time()
    encoder = ClearEncoder(s, device)
    bank = ConceptBank.load(s)
    print(f"loaded encoder + bank in {time.time() - t:.1f}s | bank: {bank.info.n_concepts:,} x {bank.info.embed_dim} "
          f"(raw file unit-norm: {bank.info.prenormalised})")

    labels = chexpert_positive_labels(s.paths.mimic_root / s.dataset.chexpert_csv)
    decoded = [load_record_image(r, s) for r in records]
    t = time.time()
    feats = encoder.encode_batch([d for d, _ in decoded])
    scores = score_concepts_batch(feats, bank, device, s.concept_bank.score_batch_size)
    print(f"encoded + scored {len(records)} image(s) in {time.time() - t:.2f}s\n")

    results = []
    for (dec, meta), row in zip(decoded, scores):
        hits = top_k(row, bank, k)
        pos = labels.get((int(meta.subject_id), int(meta.study_uid)), [])
        print("=" * 100)
        print(f"{meta.split}  study {meta.study_uid}  image {meta.image_id}  view {meta.view_position}  {dec.width}x{dec.height}")
        print(f"CheXpert positives for this study: {pos or 'none listed'}")
        print(f"score range over 368,294 concepts: min {row.min():.3f}  max {row.max():.3f}  mean {row.mean():.3f}")
        for h in hits:
            print(f"  {h.score:.4f}  [{h.concept_id:>6}]  {clean_text(h.text)}")
        results.append({"meta": meta.__dict__, "chexpert_positive": pos,
                        "top_concepts": [{"id": h.concept_id, "score": h.score, "text": clean_text(h.text)} for h in hits]})

    out = s.paths.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_top_concepts.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[written] {out / 'demo_top_concepts.json'}")


if __name__ == "__main__":
    main()
