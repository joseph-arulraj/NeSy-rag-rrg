"""Hierarchy violations on validate, the same three numbers for every model: per is_a edge the number of
studies with P(child) > P(parent) (strict, 1e-12 tolerance), the total over edges, and the number of
studies with at least one violation. Reads predictions_non_test.parquet of each run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import kg  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("hierarchy-violations", vars(args), args.run_dir) as run:
        edges = kg.load_findings()["is_a"]
        rows = []
        for spec in args.runs:
            name, rd = spec.split("=", 1)
            p = pd.read_parquet(Path(rd) / "predictions_non_test.parquet")
            v = p[p.split == "val"]
            any_v = pd.Series(False, index=v.index)
            r = {"model": name, "n_val": len(v)}
            for e in edges:
                bad = v[f"p_{e['child']}"] > v[f"p_{e['parent']}"] + 1e-12
                r[f"{e['id']} {e['child']}<{e['parent']}"] = int(bad.sum())
                any_v |= bad
            r["total_over_edges"] = sum(r[k] for k in r if k.startswith("isa_"))
            r["studies_with_any_violation"] = int(any_v.sum())
            rows.append(r)
        d = pd.DataFrame(rows).set_index("model").T
        d.to_csv(run.dir / "hierarchy_violations.csv")
        pd.set_option("display.width", 200)
        run.log("validate hierarchy violations:\n" + d.to_string())


if __name__ == "__main__":
    main()
