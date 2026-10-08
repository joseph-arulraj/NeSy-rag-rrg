"""Stage 0: patient-level MIMIC splits. FROZEN as v1 on 2026-10-06 (user confirmed); data/splits/mimic_splits_v1.parquet
is read-only and must not be regenerated. This script only reproduces the reviewed v0-DRAFT assignment.

Starts from the official MIMIC split, which is already patient-disjoint:

  official test      -> test     (in-domain test metrics only)
  official validate  -> val      (model selection / stage comparisons)
  official train     -> train, calib, thresh, heldout_loc

  * heldout_loc: every official-train patient who has an MS-CXR image or is in the Chest ImaGenome
    gold set. These are excluded from every split that fits anything (train, calib, thresh) and are
    kept for localisation evaluation. Excluded patients who fall in official validate/test stay
    there (val/test fit nothing; they are flagged in the split file).
  * calib / thresh: two disjoint patient-level samples of the remaining official-train patients,
    CALIB_FRAC and THRESH_FRAC each, chosen by a salted sha256 of subject_id, so the assignment is
    deterministic and does not depend on row order. Calibrators, thresholds and test metrics
    therefore use three different splits.

VinDr-CXR test and PadChest-GR are external test sets; they are not in this file and never get a
fitting role. Output: data/splits/mimic_splits_<VERSION>.parquet (one row per patient) + a JSON
summary. A file whose name contains DRAFT must not be treated as frozen.

  python -u scripts/build_splits.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

VERSION = "v0-DRAFT"
SALT = "nesy-cxr-split-v0"
CALIB_FRAC = 0.03
THRESH_FRAC = 0.03


def unit_hash(subject_id: int) -> float:
    """Deterministic number in [0, 1) per patient."""
    h = hashlib.sha256(f"{SALT}:{subject_id}".encode()).hexdigest()
    return int(h[:15], 16) / 16**15


def assign(official: str, excluded: bool, u: float) -> str:
    if official == "test":
        return "test"
    if official == "validate":
        return "val"
    if official != "train":
        raise ValueError(f"unknown official split {official!r}")
    if excluded:
        return "heldout_loc"
    if u < CALIB_FRAC:
        return "calib"
    if u < CALIB_FRAC + THRESH_FRAC:
        return "thresh"
    return "train"


def build(run: Run) -> pd.DataFrame:
    mimic = pd.read_parquet(P.MANIFESTS / "mimic.parquet", columns=["subject_id", "study_id", "dicom_id", "source_split",
                                                                    "is_frontal"])
    ms = pd.read_parquet(P.MANIFESTS / "mscxr.parquet", columns=["subject_id"])
    gold = pd.read_parquet(P.MANIFESTS / "imagenome_gold.parquet", columns=["subject_id"])
    run.log(f"in: {len(mimic):,} MIMIC images, {mimic.subject_id.nunique():,} patients")

    pat = mimic.groupby("subject_id").source_split.agg(["nunique", "first"])
    if (pat["nunique"] > 1).any():
        raise ValueError(f"{(pat['nunique'] > 1).sum()} patients in more than one official split")
    pat = pat.rename(columns={"first": "official_split"}).drop(columns="nunique")
    ms_set, gold_set = set(ms.subject_id), set(gold.subject_id)
    pat["is_mscxr_patient"] = pat.index.isin(ms_set)
    pat["is_imagenome_gold_patient"] = pat.index.isin(gold_set)
    pat["excluded_from_fitting"] = pat.is_mscxr_patient | pat.is_imagenome_gold_patient
    run.log(f"exclusion set: MS-CXR {len(ms_set):,} + gold {len(gold_set):,} patients, overlap "
            f"{len(ms_set & gold_set):,}, union {pat.excluded_from_fitting.sum():,}")
    pat["hash_u"] = [unit_hash(s) for s in pat.index]
    pat["split"] = [assign(o, e, u) for o, e, u in zip(pat.official_split, pat.excluded_from_fitting, pat.hash_u)]
    pat["split_version"] = VERSION
    pat = pat.reset_index()

    stats = mimic.merge(pat[["subject_id", "split"]], on="subject_id")
    summary = {}
    for sp, g in stats.groupby("split"):
        fr = g[g.is_frontal]
        summary[sp] = {"patients": int(g.subject_id.nunique()), "studies": int(g.study_id.nunique()),
                       "images": int(len(g)), "frontal_images": int(len(fr)), "studies_with_frontal": int(fr.study_id.nunique()),
                       "excluded_patients_inside": int(pat[(pat.split == sp) & pat.excluded_from_fitting].shape[0])}
        run.log(f"  {sp:12s} " + ", ".join(f"{k} {v:,}" for k, v in summary[sp].items()))
        run.metric(split=sp, **summary[sp])
    return pat, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    cfg = {"version": VERSION, "salt": SALT, "calib_frac": CALIB_FRAC, "thresh_frac": THRESH_FRAC}
    with Run("splits", cfg, args.run_dir) as run:
        pat, summary = build(run)
        P.SPLITS.mkdir(parents=True, exist_ok=True)
        out = P.SPLITS / f"mimic_splits_{VERSION}.parquet"
        if out.exists():
            old = pd.read_parquet(out)
            same = old[["subject_id", "split"]].equals(pat[["subject_id", "split"]])
            run.log(f"{out.name} exists; identical assignment: {same}")
            if not same:
                raise RuntimeError(f"refusing to overwrite {out} with a different assignment; bump VERSION instead")
        pat.to_parquet(out, index=False)
        (P.SPLITS / f"mimic_splits_{VERSION}.json").write_text(json.dumps({**cfg, "summary": summary}, indent=2))
        run.log(f"wrote {out} ({len(pat):,} patients). STATUS: DRAFT, not frozen until the user confirms.")


if __name__ == "__main__":
    main()
