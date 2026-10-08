"""MLP head ablation (user item A, 2026-10-07). Ablation only: the linear factorised head stays the main model.

Exactly the inputs of the given linear head run (same concepts file, same --extra groups, same NESY_ env),
the same fit split, labels and factorised hierarchy outputs (P(finding | parent = 1) on rows whose parent
is positive, marginals multiplied down the is_a tree, so violations are impossible by construction), and
the same Platt scaling on calib. The model per finding is a one-hidden-layer MLP (ReLU), AdamW with weight
decay, early stopping on --es-split (val as requested by the user; this makes the val numbers optimistic
for the MLP, so --es-split inner, the fit split's inner holdout as used for the linear head's C, is also
run as the unbiased comparison).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from nesy import data as D, kg, paths as P  # noqa: E402
from nesy.evaluate import binary_metrics  # noqa: E402
from nesy.runlog import Run  # noqa: E402
from probe import build_features, inner_holdout  # noqa: E402
from stage3_factorised import aw_mask, concept_features  # noqa: E402


def fit_mlp(X, y, tr, es, hidden, wd, run, label, max_epochs=40, patience=3, seed=0):
    import torch
    torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    tri, esi = np.nonzero(tr)[0], np.nonzero(es)[0]
    best, best_state, bad, hist = np.inf, None, 0, []
    g = np.random.default_rng(seed)
    for ep in range(max_epochs):
        m.train()
        for b in np.array_split(g.permutation(tri), max(1, len(tri) // 2048)):
            opt.zero_grad()
            lossf(m(Xt[b]).squeeze(1), yt[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            l = float(lossf(m(Xt[esi]).squeeze(1), yt[esi]))
        hist.append(round(l, 5))
        if l < best - 1e-5:
            best, best_state, bad = l, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    run.log(f"  {label:42s} hidden {hidden}: stopped after {len(hist)} epochs, best ES loss {best:.4f} at epoch {int(np.argmin(hist))}")
    return m, int(np.argmin(hist)), hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-run", required=True, help="the linear head run whose inputs are reused")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--es-split", choices=["val", "inner"], default="val")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run(f"stage3-mlp-h{args.hidden}-es{args.es_split}", vars(args), args.run_dir) as run:
        import torch
        torch.set_num_threads(args.threads)
        lin = json.loads((Path(args.linear_run) / "config.json").read_text())
        run.log(f"inputs of {args.linear_run}: concepts {lin['concepts']}, extra '{lin['extra']}', env {lin.get('env_nesy')}")
        t = D.study_table()
        fit = (t.split == "train").values
        Xc, names = concept_features(t, fit, run, Path(lin["concepts"]) if Path(lin["concepts"]).is_absolute() else P.V2 / lin["concepts"])
        blocks = [Xc]
        for g in [x for x in lin["extra"].split(",") if x]:
            Xg, ng = build_features(t, [g], fit, run)
            blocks.append(Xg)
            names += ng
        X = np.hstack(blocks).astype(np.float32)
        run.log(f"design {X.shape}")
        inner = inner_holdout(t.subject_id.values) & fit
        core = fit & ~inner
        val = (t.split == "val").values
        cal = (t.split == "calib").values
        cond, info = {}, {}
        t0 = time.time()
        for f in kg.topo_order()[::-1]:
            y = t[f].values
            parents = kg.parents_of(f)
            known = ~np.isnan(y)
            gate = (t[parents[0]].values == 1) if parents else np.ones(len(t), bool)
            label = f"P({f} | {parents[0]}=1)" if parents else f"P({f})"
            Xf = aw_mask(X, names, f)
            tr = (core if args.es_split == "inner" else fit) & known & gate
            es = (inner if args.es_split == "inner" else val) & known & gate
            m, ep, hist = fit_mlp(Xf, np.nan_to_num(y), tr, es, args.hidden, args.weight_decay, run, label)
            with torch.no_grad():
                s = m(torch.from_numpy(Xf)).squeeze(1).numpy()
            cm = cal & known & gate
            platt = LogisticRegression(C=1e6).fit(s[cm, None], np.nan_to_num(y[cm]).astype(int))
            cond[f] = platt.predict_proba(s[:, None])[:, 1]
            info[f] = {"best_epoch": ep, "es_loss": hist}
            torch.save(m.state_dict(), run.dir / f"mlp_{f}.pt")
        marg = {}
        for f in kg.topo_order()[::-1]:
            ps = kg.parents_of(f)
            marg[f] = cond[f] * (marg[ps[0]] if ps else 1.0)
        viol = {e["id"]: int((marg[e["child"]][val] > marg[e["parent"]][val] + 1e-12).sum()) for e in kg.load_findings()["is_a"]}
        rows = [{"finding": f, "split": "val", **binary_metrics(t[f].values[val], marg[f][val])} for f in D.FINDINGS]
        v = pd.DataFrame(rows).set_index("finding")
        v.to_csv(run.dir / "val_metrics.csv")
        (run.dir / "training.json").write_text(json.dumps(info, indent=1))
        out = pd.DataFrame({"study_id": t.study_id, "split": t.split, **{f"p_{f}": marg[f] for f in D.FINDINGS}})
        out[out.split != "test"].to_parquet(run.dir / "predictions_non_test.parquet", index=False)
        b = pd.read_csv(Path(args.linear_run) / "val_metrics.csv").set_index("finding")
        cmp = pd.DataFrame({"auroc_linear": b.auroc, "auroc_mlp": v.auroc, "d_auroc": v.auroc - b.auroc,
                            "ece_linear": b.ece, "ece_mlp": v.ece, "d_ece": v.ece - b.ece})
        cmp.to_csv(run.dir / "val_vs_linear.csv")
        run.log(f"hierarchy violations on val: {viol} -> total {sum(viol.values())}")
        run.log("VAL MLP vs linear head:\n" + cmp.round(4).to_string())
        run.log(f"macro: AUROC {v.auroc.mean():.4f} (linear {b.auroc.mean():.4f}, d {cmp.d_auroc.mean():+.4f}); "
                f"ECE {v.ece.mean():.4f} (linear {b.ece.mean():.4f}); {time.time() - t0:.0f} s")
        run.metric(summary="val_macro", auroc=float(v.auroc.mean()), ece=float(v.ece.mean()), hierarchy_violations=int(sum(viol.values())))


if __name__ == "__main__":
    main()
