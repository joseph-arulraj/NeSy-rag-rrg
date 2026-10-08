"""Label counts for the external test sets, to support review of kg/external_maps/*.yaml (user request
2026-10-07). Reads ONLY the external test labels; no predictions, no metrics, nothing trained or selected.

Per dataset:
  raw_label_counts.csv   every source label: images positive, which of our findings it maps to (from-lists), or unmapped
  finding_counts.csv     per our finding: positives under the draft mapping, direct and after is_a closure
                         (a child positive makes its ancestors positive), and the evaluable flag
  only_unmapped          images whose only positive labels are unmapped (neither normal nor positive for any finding)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nesy import kg  # noqa: E402
from nesy.runlog import Run  # noqa: E402

D = Path("/scratch/prj/bhi_zihe_imaging/mimic_cxr_full")
MAPS = Path(__file__).resolve().parents[1] / "kg/external_maps"


def vindr_labels() -> dict[str, set[str]]:
    t = pd.read_csv(D / "VinDr_cxr/annotations/image_labels_test.csv")
    lab = [c for c in t.columns if c != "image_id"]
    return {r.image_id: {c for c in lab if r[c] == 1} for _, r in t.iterrows()}


def padchest_labels() -> dict[str, set[str]]:
    d = json.load(open(D / "PadChest_GR/grounded_reports_20240819.json"))
    out = {}
    for s in d:
        labs = {lab for f in s["findings"] for lab in (f.get("labels") or [])}
        if not any(f.get("abnormal") for f in s["findings"]):
            labs.add("Normal")       # study with no abnormal sentence
        out[s["ImageID"]] = labs
    return out


def count(name, labels: dict[str, set[str]], mp: dict, run: Run):
    fmap = mp["findings"]
    nf = {x.lower() for x in mp.get("no_finding", {}).get("from", [])}
    by_label = {}
    for labs in labels.values():
        for x in labs:
            by_label[x] = by_label.get(x, 0) + 1
    rows = []
    for x, n in sorted(by_label.items(), key=lambda kv: -kv[1]):
        to = [f for f, e in fmap.items() if x.lower() in {y.lower() for y in e.get("from", [])}]
        rows.append({"label": x, "images_positive": n, "maps_to": ", ".join(to) if to else ("no_finding" if x.lower() in nf else "UNMAPPED")})
    raw = pd.DataFrame(rows)
    raw.to_csv(run.dir / f"{name}_raw_label_counts.csv", index=False)
    pos_direct = {f: 0 for f in fmap}
    pos_closed = {f: 0 for f in fmap}
    only_unmapped = 0
    for labs in labels.values():
        low = {x.lower() for x in labs}
        direct = {f for f, e in fmap.items() if low & {y.lower() for y in e.get("from", [])}}
        closed = set(direct) | {a for f in direct for a in kg.ancestors(f) if a in fmap}
        for f in direct:
            pos_direct[f] += 1
        for f in closed:
            pos_closed[f] += 1
        if labs and not direct and not (low & nf):
            only_unmapped += 1
    fc = pd.DataFrame([{"finding": f, "evaluable": fmap[f].get("evaluable", True), "positives_direct": pos_direct[f],
                        "positives_with_is_a": pos_closed[f]} for f in fmap])
    fc.to_csv(run.dir / f"{name}_finding_counts.csv", index=False)
    n_nf = sum(1 for labs in labels.values() if {x.lower() for x in labs} & nf)
    run.log(f"{name}: {len(labels):,} images; no_finding-mapped {n_nf:,}; only unmapped labels {only_unmapped:,}; "
            f"no label at all {sum(1 for v in labels.values() if not v):,}")
    run.log(f"{name} per finding:\n" + fc.to_string(index=False))
    run.log(f"{name} raw labels:\n" + raw.to_string(index=False))
    return {"images": len(labels), "no_finding": n_nf, "only_unmapped": only_unmapped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("external-label-counts", vars(a), a.run_dir) as run:
        summ = {}
        summ["vindr_test"] = count("vindr_test", vindr_labels(), yaml.safe_load((MAPS / "vindr_to_findings.yaml").read_text()), run)
        summ["padchest_gr"] = count("padchest_gr", padchest_labels(), yaml.safe_load((MAPS / "padchest_gr_to_findings.yaml").read_text()), run)
        (run.dir / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
