"""KG task 2: compare our is_a edges (findings, and device type -> support_devices) with RadLex's is_a nesting
of the matched terms (kg/radlex_map.yaml). Read-only: our hierarchy is not changed.
Per edge: agree (the parent's RadLex class is an is_a ancestor of the child's), differ (both matched, not an
ancestor; the RadLex parent chains are shown), not comparable (child or parent has no RadLex class).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import kg  # noqa: E402
from nesy.runlog import Run  # noqa: E402

RADLEX = Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/data/radlex/raw/Radlex.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("compare-hierarchy", vars(a), a.run_dir) as run:
        r = pd.read_csv(RADLEX, usecols=["Class ID", "Preferred Label", "Parents"], dtype=str, low_memory=False)
        r["rid"] = r["Class ID"].str.extract(r"(RID\d+)")
        r = r.drop_duplicates("rid").set_index("rid")
        par = {rid: [x for x in str(p).split("|") if "RID" in x] for rid, p in r.Parents.fillna("").items()}
        par = {k: [v.rsplit("/", 1)[-1] for v in vs] for k, vs in par.items()}
        lab = r["Preferred Label"].to_dict()

        def ancestors(rid):
            seen, stack = [], [rid]
            while stack:
                x = stack.pop()
                for p in par.get(x, []):
                    if p not in seen:
                        seen.append(p)
                        stack.append(p)
            return seen

        def chain(rid, n=6):
            out, x = [], rid
            for _ in range(n):
                ps = par.get(x, [])
                if not ps:
                    break
                x = ps[0]
                out.append(f"{x} {lab.get(x, '')}")
            return " > ".join(out)
        m = {(e["kind"], e["key"]): e for e in kg.load_radlex()["entries"]}
        edges = [(e["id"], e["child"], e["parent"]) for e in kg.load_findings()["is_a"]]
        rows = []
        for eid, c, p in edges:
            rows.append((eid, "finding", c, p, m[("finding", c)], m[("finding", p)]))
        for d in kg.load_devices()["device_types"]:
            rows.append(("device", "device", d["id"], d["finding"], m[("device", d["id"])], m[("finding", d["finding"])]))
        out = []
        for eid, kind, c, p, mc, mp in rows:
            rc, rp = mc.get("radlex_id"), mp.get("radlex_id")
            if not rc or not rp:
                verdict = "not comparable"
                detail = f"no RadLex class for {'child' if not rc else 'parent'}"
            elif rc == rp:
                verdict, detail = "differ", "child and parent map to the same RadLex class"
            elif rp in ancestors(rc):
                verdict, detail = "agree", f"RadLex: {chain(rc)}"
            else:
                verdict, detail = "differ", f"RadLex child chain: {chain(rc)}; parent {rp} {lab.get(rp)} chain: {chain(rp)}"
            out.append({"edge": eid, "child": c, "parent": p, "child_rid": rc, "child_match": mc["match"], "parent_rid": rp,
                        "parent_match": mp["match"], "verdict": verdict, "detail": detail})
        d = pd.DataFrame(out)
        d.to_csv(run.dir / "hierarchy_comparison.csv", index=False)
        run.log("verdicts: " + str(d.verdict.value_counts().to_dict()))
        for x in out:
            run.log(f"  {x['edge']:8s} {x['child']:26s} is_a {x['parent']:26s} -> {x['verdict']:15s} {x['detail'][:230]}")


if __name__ == "__main__":
    main()
