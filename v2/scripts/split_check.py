"""Item 2: positives per finding in the frozen calib and thresh splits (studies with a frontal image,
i.e. the studies the models score). Flags findings with < 50 positives. Reports only; never re-splits."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from nesy import kg, manifests as M, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

MIN_POS = 50

with Run("split-check", {"min_pos": MIN_POS, "split_version": M.SPLIT_VERSION}) as run:
    lab = pd.read_parquet(P.V2 / "data/labels/labels_v1.parquet")
    fr = M.load("mimic", columns=["subject_id", "study_id", "is_frontal"])
    with_frontal = set(fr[fr.is_frontal].study_id)
    lab = lab[lab.study_id.isin(with_frontal)]
    rows = []
    for sp in ["train", "calib", "thresh", "val", "test"]:
        g = lab[lab.split == sp]
        for f in kg.finding_ids() + ["no_finding"]:
            rows.append({"split": sp, "finding": f, "studies": len(g), "pos": int((g[f] == 1).sum()),
                         "neg": int((g[f] == 0).sum()), "masked": int(g[f].isna().sum())})
    t = pd.DataFrame(rows)
    piv = t.pivot(index="finding", columns="split", values="pos")[["train", "calib", "thresh", "val", "test"]]
    run.log("positives per finding (studies with a frontal image):\n" + piv.to_string())
    flags = t[t.split.isin(["calib", "thresh"]) & (t.pos < MIN_POS)]
    run.log(f"FLAGGED (< {MIN_POS} positives in calib or thresh):\n" + (flags.to_string(index=False) if len(flags) else "none"))
    t.to_csv(run.dir / "positives_per_split.csv", index=False)
    for r in flags.itertuples():
        run.metric(flag=r.finding, split=r.split, pos=r.pos)
