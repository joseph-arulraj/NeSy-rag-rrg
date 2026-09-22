#!/usr/bin/env python
"""Show what FAISS retrieval returns: for a few query images, the top-k similar TRAIN studies and their reports.

  python scripts/run_retrieval_demo.py --query-split test                 # patient-disjoint from train: no exclusion needed
  python scripts/run_retrieval_demo.py --query-split train                # own patient is excluded from the results
  python scripts/run_retrieval_demo.py --query-split train --no-exclude   # WITHOUT exclusion: the image retrieves itself (leakage)
  python scripts/run_retrieval_demo.py --query-split validate --config configs/hpc.yaml

Develop and tune on `validate`; the `test` split is for the final evaluation only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402
from rrg.core.runtime import configure_runtime  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.datasets.reports import ReportStore  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402
from rrg.retrieval.faiss_index import FaissRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--query-split", default=None, help="split to draw query images from (default: demo.split)")
    ap.add_argument("--num", type=int, default=None, help="number of queries (default: demo.num_images)")
    ap.add_argument("--no-exclude", action="store_true", help="disable same-patient exclusion (to see the leak)")
    ap.add_argument("--report-chars", type=int, default=260)
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    device = configure_runtime(s)
    split = args.query_split or s.demo.split
    encoder = ClearEncoder(s, device)
    retriever = FaissRetriever.load(s, expected_checkpoint_sha256=encoder.checkpoint_sha256,
                                    exclude_same_patient=False if args.no_exclude else None)
    m = retriever.manifest
    print(f"index '{m.name}': {m.n_vectors:,} x {m.embed_dim} {m.precision} vectors, splits={m.corpus_splits}, "
          f"k={retriever.k}, exclude_same_patient={retriever.exclude_same_patient}"
          + (f", SMOKE INDEX (first {m.built_limit} studies)" if m.built_limit else ""))

    records, _ = MimicCxrIndex.from_settings(s).study_records(split)
    records = records[: args.num or s.demo.num_images]
    if not records:
        sys.exit(f"no query images in split {split!r}")
    reports = ReportStore.from_settings(s)
    for rec in records:
        dec, meta = load_record_image(rec, s)
        emb = encoder.encode(dec)
        hits = retriever.search_embedding(emb, subject_id=rec.subject_id)
        print("=" * 110)
        print(f"QUERY [{split}] subject {rec.subject_id} study {rec.study_id} ({rec.view_position})")
        print("  report:", reports.get_clean(rec.subject_id, rec.study_id)[: args.report_chars])
        for h in hits:
            flag = "  <-- SAME IMAGE" if h.dicom_id == rec.dicom_id else ("  <-- SAME PATIENT" if h.subject_id == rec.subject_id else "")
            print(f"  #{h.rank}  cos {h.similarity:.4f}  subject {h.subject_id} study {h.study_id}{flag}")
            print("      ", reports.get_clean(h.subject_id, h.study_id)[: args.report_chars])
        if not hits:
            print("  (no neighbours above the similarity floor)")


if __name__ == "__main__":
    main()
