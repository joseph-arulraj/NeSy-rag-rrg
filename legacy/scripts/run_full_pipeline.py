#!/usr/bin/env python
"""End-to-end: B + C + D -> calibration/fusion (N23) -> initial belief graph (N24) -> N25 ->
verified belief graph (N26) -> N27 LLM -> draft report (N28), for N images. pipeline.md's full
Stage 1-3 chain, v1 (single-pass, no A, no regeneration loop).

Prerequisites (each a separate script, run once, in this order):
  1. scripts/build_concept_tags.py   -- fresh concept tags (Evidence B)
  2. scripts/build_faiss_index.py    -- retrieval index (Evidence D)
  3. scripts/train_cbm.py            -- CBM head (Evidence C)
  4. scripts/run_calibration.py      -- B/C calibrators + thresholds (needs 1 and 3 done first)

  export RRG_LLM_API_KEY=sk-...
  python scripts/run_full_pipeline.py --split test --num-images 5
"""
from __future__ import annotations

import os

# See scripts/run_calibration.py's comment -- must be set before radgraph/transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.cbm.concepts import load_or_resolve_concept_ids  # noqa: E402
from rrg.cbm.infer import CBMPredictor  # noqa: E402
from rrg.cbm.model import load_head  # noqa: E402
from rrg.concepts.bank import ConceptBank, load_tags  # noqa: E402
from rrg.concepts.grouping import group_and_rank  # noqa: E402
from rrg.concepts.similarity import score_concepts_batch  # noqa: E402
from rrg.concepts.tagging.finding_vocab import load_finding_vocabulary  # noqa: E402
from rrg.core.config import load_settings  # noqa: E402
from rrg.core.runtime import configure_runtime  # noqa: E402
from rrg.core.types import ImageEmbedding, StudyMeta  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.datasets.reports import ReportStore  # noqa: E402
from rrg.fusion.calibrate import load_calibration  # noqa: E402
from rrg.fusion.fuse import fuse_study  # noqa: E402
from rrg.fusion.graph import build_initial_graph  # noqa: E402
from rrg.fusion.rules.radlex_client import RadLexClient  # noqa: E402
from rrg.fusion.rules.rule_engine import apply_rules  # noqa: E402
from rrg.generation.llm_client import LLMClient  # noqa: E402
from rrg.generation.report import generate_draft_report  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402
from rrg.retrieval.evidence_d import build_evidence_d  # noqa: E402
from rrg.retrieval.faiss_index import FaissRetriever  # noqa: E402
from rrg.retrieval.radgraph_parser import RadGraphParser  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-images", type=int, default=5)
    ap.add_argument("--out", default=None, help="per-study JSONL output (default: <output_dir>/full_pipeline/<split>.jsonl)")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    device = configure_runtime(s)

    # ---- shared resources, loaded once
    bank = ConceptBank.load(s).with_tags(
        load_tags(s.paths.model_weights_dir / "concept_tags.jsonl.gz", s.concept_bank.expected_n_concepts)
    )
    grouping_index = bank.grouping_index()
    radlex = RadLexClient.load(
        s.radlex.snapshot_path, s.radlex.chest_scope_root_label, s.radlex.version, tuple(s.radlex.additional_scope_roots)
    )
    vocab = load_finding_vocabulary(s.tagging.finding_synonyms_path)
    radgraph = RadGraphParser(model_type=s.radgraph.model_type, device=s.radgraph.device, batch_size=s.radgraph.batch_size)
    reports = ReportStore.from_settings(s)
    encoder = ClearEncoder(s, device)

    concept_ids = load_or_resolve_concept_ids(s.cbm.concepts_path, s.cbm.concept_ids_cache, bank.concepts)
    head, head_info = load_head(s.cbm.head_checkpoint)
    cbm = CBMPredictor(head, head_info, concept_ids, bank)

    retriever = FaissRetriever.load(s)
    calib = load_calibration(s)
    llm = LLMClient(s)

    index = MimicCxrIndex.from_settings(s)
    records, _ = index.study_records(args.split)
    records = records[: args.num_images]
    if not records:
        sys.exit(f"no images in split {args.split!r}")

    out_path = Path(args.out) if args.out else s.paths.output_dir / "full_pipeline" / f"{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            print(f"[{i + 1}/{len(records)}] {rec.dicom_id}")
            decoded, _meta = load_record_image(rec, s)
            feat_t = encoder.encode_batch([decoded])[0]
            feat = feat_t.numpy()
            raw_scores = score_concepts_batch(feat[None, :], bank, device, s.concept_bank.score_batch_size)[0]

            # ---- B
            b_groups = group_and_rank(raw_scores, bank, s.grouping, index=grouping_index)

            # ---- C
            c_preds = cbm.predict(raw_scores)

            # ---- D
            embedding = ImageEmbedding(
                vector=feat.astype("float32"), embed_dim=feat.shape[0], checkpoint=encoder.checkpoint_name,
                checkpoint_sha256=encoder.checkpoint_sha256, normalised=True, precision="fp32",
            )
            neighbours = retriever.search_embedding(embedding, subject_id=rec.subject_id)
            d_items = build_evidence_d(neighbours, reports, radgraph, radlex, vocab, s)

            # ---- N23 fuse -> N24 initial graph -> N25 -> N26 verified graph
            fused = fuse_study(b_groups, c_preds, d_items, calib, s)
            study_meta = StudyMeta(
                study_uid=str(rec.study_id), image_id=rec.dicom_id, subject_id=str(rec.subject_id),
                split=rec.split, view_position=rec.view_position, patient_orientation=rec.patient_orientation,
                spatial_unavailable=not s.spatial_enabled,
            )
            initial_graph = build_initial_graph(fused, study_meta)
            verified_graph = apply_rules(initial_graph, s, log=print)

            # ---- N27 -> N28
            draft = generate_draft_report(verified_graph, llm)

            row = {
                "dicom_id": rec.dicom_id, "study_id": rec.study_id, "subject_id": rec.subject_id,
                "n_findings_admitted": len(verified_graph.findings),
                "n_findings_rejected": len(verified_graph.rejected),
                "n_rule_applications": len(verified_graph.audit),
                "findings": [
                    {"finding_id": f.finding_id, "label": f.label, "anatomy": f.anatomy, "laterality": f.laterality.value,
                     "polarity": f.polarity, "confidence": f.confidence,
                     "support": [asdict(ref) for ref in f.support]}
                    for f in verified_graph.findings
                ],
                "rejected_findings": [
                    {"finding_id": f.finding_id, "label": f.label, "anatomy": f.anatomy, "laterality": f.laterality.value,
                     "polarity": f.polarity, "confidence": f.confidence}
                    for f in verified_graph.rejected
                ],
                "rule_audit": [asdict(r) for r in verified_graph.audit],
                "draft_report": draft.text,
                "llm_model": draft.model,
                "prompt_hash": draft.prompt_hash,
            }
            fh.write(json.dumps(row) + "\n")
            print(f"  {len(verified_graph.findings)} findings, {len(verified_graph.audit)} rule application(s)")
            print(f"  --- draft report ---\n{draft.text}\n")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
