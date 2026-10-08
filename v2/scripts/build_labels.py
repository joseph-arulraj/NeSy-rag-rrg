"""Item 1: study-level label table from the official CheXpert labels.

Rules, applied in this order, each counted:
  R1  blank (not mentioned)          -> 0 (negative)
  R2  -1 (uncertain)                 -> masked (NaN, excluded from the loss)
  R3  positive child                 -> every ancestor positive (kg/findings.yaml is_a, transitive)
  R4  uncertain child, blank parent  -> parent masked; cascades upward (a parent masked by R4 counts
                                        as uncertain for its own parent). Explicit 0 parents are kept.
  R0  study without a CheXpert row (8 unlabelable studies) -> all masked
any_abnormality (KG root, no CheXpert column) starts as blank -> 0 and is set by the same rules: R3
makes it 1 when any pathology is positive; R4 masks it when a pathology is uncertain/masked and none is
positive. Studies without a CheXpert row are masked. Support devices are not a child, so they never
count. Agreement of (any_abnormality == 0) with CheXpert's own No Finding is reported.

Output: data/labels/labels_v1.parquet (one row per study; findings as float 1/0/NaN, plus `src_<f>`
= original CheXpert value code: 1 pos, 0 neg, -1 unc, 9 blank, -9 no row).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import kg  # noqa: E402
from nesy import manifests as M  # noqa: E402
from nesy import paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

OUT = P.V2 / "data/labels/labels_v2.parquet"   # v2: adds the any_abnormality root (2026-10-06)


def build(run: Run) -> pd.DataFrame:
    kgd = kg.load_findings()
    F = kg.finding_ids()
    col = {f["id"]: "cx_" + f["chexpert"].lower().replace(" ", "_") for f in kgd["findings"] if f.get("chexpert")}
    m = M.load("mimic", columns=["subject_id", "study_id", "has_chexpert_row", "cx_no_finding", *col.values()])
    st = m.drop_duplicates("study_id").set_index("study_id")
    run.log(f"in: {len(st):,} studies ({(~st.has_chexpert_row).sum()} without a CheXpert row)")

    src = pd.DataFrame(index=st.index)
    val = pd.DataFrame(index=st.index, dtype=float)
    counts = []
    for f in F:
        raw = st[col[f]] if f in col else pd.Series(np.nan, index=st.index)     # KG-only node: starts blank
        code = np.where(~st.has_chexpert_row, -9, np.where(raw.isna(), 9, raw.fillna(0))).astype(int)
        src[f] = code
        v = np.select([code == 1, code == 0, code == 9, code == -1, code == -9], [1.0, 0.0, 0.0, np.nan, np.nan])
        val[f] = v
        counts.append({"finding": f, "R1_blank_to_neg": int((code == 9).sum()), "R2_uncertain_masked": int((code == -1).sum()),
                       "R0_no_row_masked": int((code == -9).sum())})
    cnt = pd.DataFrame(counts).set_index("finding")
    cnt["R3_pos_from_blank"] = 0
    cnt["R3_pos_from_explicit_neg"] = 0
    cnt["R3_pos_from_uncertain"] = 0
    cnt["R4_masked"] = 0

    order = kg.topo_order()
    # R3: positive propagation, children first
    for f in order:
        pos = val[f] == 1
        for p in kg.parents_of(f):
            change = pos & (val[p] != 1)
            for name, code in [("R3_pos_from_blank", 9), ("R3_pos_from_explicit_neg", 0), ("R3_pos_from_uncertain", -1)]:
                cnt.loc[p, name] += int((change & (src[p] == code)).sum())
            val.loc[change, p] = 1.0
    # R4: uncertainty masking, children first; masked-by-R4 parents cascade
    unc = pd.DataFrame({f: src[f] == -1 for f in F})
    for f in order:
        for p in kg.parents_of(f):
            change = unc[f] & (src[p] == 9) & (val[p] == 0)
            cnt.loc[p, "R4_masked"] += int(change.sum())
            val.loc[change, p] = np.nan
            unc.loc[change, p] = True

    ab = val["any_abnormality"]
    cx_nf = st["cx_no_finding"]
    both = ab.notna()
    agree = ((ab == 0) == (cx_nf == 1))[both].mean()
    run.log(f"any_abnormality: 1 {int((ab == 1).sum()):,}, 0 {int((ab == 0).sum()):,}, masked {int(ab.isna().sum()):,}; "
            f"agreement of (any_abnormality == 0) with CheXpert 'No Finding' = 1 where defined: {agree:.3%}")
    dis = ab.eq(0) & cx_nf.ne(1)
    run.log(f"  any_abnormality = 0 but CheXpert No Finding blank: {int(dis.sum()):,} (CheXpert suppresses No Finding for "
            f"mentions outside the 14 labels, e.g. hyperinflation)")

    run.log("rule counts per finding:\n" + cnt.to_string())
    run.log("totals: " + ", ".join(f"{c} {int(cnt[c].sum()):,}" for c in cnt.columns))
    for f in F:
        run.metric(finding=f, **{k: int(v) for k, v in cnt.loc[f].items()},
                   pos=int((val[f] == 1).sum()), neg=int((val[f] == 0).sum()), masked=int(val[f].isna().sum()))

    # invariant: no child positive with a non-positive parent
    for e in kgd["is_a"]:
        bad = ((val[e["child"]] == 1) & (val[e["parent"]] != 1)).sum()
        if bad:
            raise AssertionError(f"hierarchy violation after closure: {e['id']} {bad}")
    run.log("hierarchy check: 0 violations")

    out = val.add_prefix("").reset_index()
    out = out.merge(st[["subject_id"]].reset_index(), on="study_id")
    for f in F:
        out["src_" + f] = src[f].values
    out = out.merge(M.load_splits()[["subject_id", "split"]], on="subject_id", how="left")
    final = pd.DataFrame({f: [(out[f] == 1).sum(), (out[f] == 0).sum(), out[f].isna().sum()] for f in F},
                         index=["pos", "neg", "masked"]).T
    run.log("final label counts (all studies):\n" + final.to_string())
    return out, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("labels", {"out": str(OUT), "kg": str(kg.KG_DIR / "findings.yaml"), "split_version": M.SPLIT_VERSION}, args.run_dir) as run:
        out, cnt = build(run)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
        cnt.to_csv(Path(run.dir) / "rule_counts.csv")
        run.log(f"wrote {OUT} ({len(out):,} studies)")


if __name__ == "__main__":
    main()
