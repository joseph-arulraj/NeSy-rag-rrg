"""Lane C items 11-13: DenseNet-121 baseline (ImageNet init) on the 256-px letterbox cache.

Same studies, splits and labels as the pipeline (study table / labels_v2: blank = negative, uncertain
masked, is_a closure). One independent sigmoid per finding, any_abnormality included (14 outputs).
Train on the fit (train) split with 224-px random crops (no horizontal flip: it would swap left and
right); evaluate with the 224-px centre crop. Masked BCE (uncertain labels contribute nothing).
Checkpoint every epoch (model, optimiser, scheduler, epoch, history) and resume from the last one.
Epoch chosen on the INNER HOLDOUT of the fit split (5% of fit patients, the same holdout the linear heads use
to choose C), which is excluded from training (user decision 2026-10-07). Validate macro AUROC is logged
every epoch only so that the best-on-validate epoch can be reported as an optimistic upper bound. Then Platt scaling per finding on calib; predictions for every
split (test included, no test metric is computed) and validate metrics (AUROC, AUPRC, ECE, hierarchy
violations).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, kg, paths as P  # noqa: E402
from nesy.evaluate import binary_metrics  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import inner_holdout  # noqa: E402

CACHE = P.V2 / "data/image_cache_256"
WEIGHTS = P.V2 / "models/densenet121-a639ec97.pth"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_cache(run):
    ids = pd.read_parquet(CACHE / "ids.parquet")
    imgs = np.zeros((len(ids), 256, 256), np.uint8)
    seen = np.zeros(len(ids), bool)
    shards = sorted(CACHE.glob("shard_*.npz"))
    t0 = time.time()
    for i, p in enumerate(shards):
        z = np.load(p)
        imgs[z["row"]] = z["img"]
        seen[z["row"]] = True
        if (i + 1) % 10 == 0:
            run.log(f"  cache: {i + 1}/{len(shards)} shards loaded, {time.time() - t0:.0f} s")
    assert seen.all(), f"cache incomplete: {int((~seen).sum())} images missing"
    run.log(f"cache loaded: {len(ids):,} images from {len(shards)} shards in {time.time() - t0:.0f} s")
    return ids, imgs


def build_model(n_out):
    import torch
    import torchvision
    m = torchvision.models.densenet121(weights=None)
    sd = torch.load(WEIGHTS, map_location="cpu")
    pat = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
    sd = {(pat.sub(r"\1\2", k) if pat.match(k) else k): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.classifier = torch.nn.Linear(m.classifier.in_features, n_out)
    return m


def batches_to_tensor(imgs, rows, train, rng, device):
    import torch
    n = len(rows)
    if train:
        ox, oy = rng.integers(0, 33, n), rng.integers(0, 33, n)
    else:
        ox = oy = np.full(n, 16)
    x = np.stack([imgs[r, y:y + 224, xx:xx + 224] for r, y, xx in zip(rows, oy, ox)])
    t = torch.from_numpy(x).to(device, non_blocking=True).float().div_(255.0)[:, None].expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=device)[None, :, None, None]
    std = torch.tensor(STD, device=device)[None, :, None, None]
    return ((t - mean) / std).contiguous(memory_format=torch.channels_last)


def predict(model, imgs, rows, device, bs=256):
    import torch
    out = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a in range(0, len(rows), bs):
            out.append(model(batches_to_tensor(imgs, rows[a:a + bs], False, None, device)).float().cpu().numpy())
    return np.concatenate(out)


def macro_auroc(y, s):
    from sklearn.metrics import roc_auc_score
    a = []
    for j in range(y.shape[1]):
        k = ~np.isnan(y[:, j])
        if len(np.unique(y[k, j])) == 2:
            a.append(roc_auc_score(y[k, j], s[k, j]))
    return float(np.mean(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-dir", default=None, help="fixed checkpoint directory, so a relaunch in a new run dir resumes")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("densenet121", {**vars(args), "weights": str(WEIGHTS), "cache": str(CACHE), "augment": "random 224 crop of 256 letterbox, no flip",
                             "eval": "centre 224 crop"}, args.run_dir) as run:
        import torch
        torch.manual_seed(args.seed)
        dev = "cuda"
        run.log(f"GPU {torch.cuda.get_device_name(0)}")
        t = D.study_table()
        ids, imgs = load_cache(run)
        assert (ids.study_id.values == t.study_id.values).all() if len(ids) == len(t) else False, "cache order differs from study table"
        F = D.FINDINGS
        Y = t[F].values.astype(np.float32)
        split = t.split.values
        inner = inner_holdout(t.subject_id.values) & (split == "train")
        tr = np.nonzero((split == "train") & ~inner)[0]
        inn, va = np.nonzero(inner)[0], np.nonzero(split == "val")[0]
        run.log(f"studies {len(t):,}; fit split {int((split == 'train').sum()):,} = trained on {len(tr):,} + inner holdout "
                f"{len(inn):,} (epoch selection, never trained on); val {len(va):,}; outputs {len(F)}: {F}")
        model = build_model(len(F)).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = len(tr) // args.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
        ck = Path(args.ckpt_dir) if args.ckpt_dir else run.dir / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        start, hist = 0, []
        last = sorted(ck.glob("epoch_*.pt"))
        if last:
            st = torch.load(last[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start, hist = st["epoch"] + 1, st["hist"]
            run.log(f"resumed from {last[-1]} (next epoch {start})")
        Yt = torch.from_numpy(Y)
        for ep in range(start, args.epochs):
            rng = np.random.default_rng(args.seed * 1000 + ep)
            order = rng.permutation(tr)
            model.train()
            t0, run_loss = time.time(), []
            for s in range(steps_per_epoch):
                rows = order[s * args.batch:(s + 1) * args.batch]
                x = batches_to_tensor(imgs, rows, True, rng, dev)
                y = Yt[rows].to(dev)
                known = ~torch.isnan(y)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(x).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[known], y[known])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                run_loss.append(loss.item())
                if (s + 1) % args.log_every == 0:
                    el = time.time() - t0
                    run.log(f"epoch {ep} step {s + 1}/{steps_per_epoch} loss {np.mean(run_loss[-args.log_every:]):.4f} "
                            f"lr {sched.get_last_lr()[0]:.2e} {((s + 1) * args.batch) / el:.0f} img/s, ETA epoch {el / (s + 1) * (steps_per_epoch - s - 1) / 60:.1f} min")
            auc_in = macro_auroc(Y[inn], predict(model, imgs, inn, dev))
            auc = macro_auroc(Y[va], predict(model, imgs, va, dev))
            hist.append({"epoch": ep, "train_loss": float(np.mean(run_loss)), "inner_macro_auroc": auc_in, "val_macro_auroc": auc,
                         "minutes": (time.time() - t0) / 60})
            run.metric(**hist[-1])
            path = ck / f"epoch_{ep:02d}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep, "hist": hist}, path)
            run.log(f"EPOCH {ep}: train loss {hist[-1]['train_loss']:.4f}; inner-holdout macro AUROC {auc_in:.4f}; "
                    f"(val {auc:.4f}, for the optimistic bound only); {hist[-1]['minutes']:.1f} min; checkpoint {path}")
        best = max(hist, key=lambda h: h["inner_macro_auroc"])
        opt_ep = max(hist, key=lambda h: h["val_macro_auroc"])
        run.log(f"MAIN: epoch chosen on the inner holdout: {best['epoch']} (inner {best['inner_macro_auroc']:.4f}); "
                f"OPTIMISTIC bound: best-on-validate epoch {opt_ep['epoch']} (val {opt_ep['val_macro_auroc']:.4f}); history {json.dumps(hist)}")
        finalize(run, model, ck, best["epoch"], run.dir, imgs, t, Y, F, split, dev, "main (inner-holdout epoch)")
        od = run.dir / "optimistic_best_on_val"
        od.mkdir(exist_ok=True)
        if opt_ep["epoch"] == best["epoch"]:
            (od / "SAME_EPOCH.txt").write_text(f"best-on-validate epoch = inner-holdout epoch = {best['epoch']}\n")
            run.log("optimistic bound: same epoch as the main model")
        else:
            finalize(run, model, ck, opt_ep["epoch"], od, imgs, t, Y, F, split, dev, "OPTIMISTIC (epoch chosen on validate)")


def finalize(run, model, ck, epoch, outdir, imgs, t, Y, F, split, dev, label):
    import torch
    if True:                                                     # body kept at the original indentation
        st = torch.load(ck / f"epoch_{epoch:02d}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        torch.save({"model": model.state_dict(), "findings": F, "epoch": epoch}, outdir / "best_model.pt")
        # logits for every study, then Platt on calib
        allrows = np.arange(len(t))
        logits = predict(model, imgs, allrows, dev)
        from sklearn.linear_model import LogisticRegression
        probs, platt = {}, {}
        cal = split == "calib"
        for j, f in enumerate(F):
            k = cal & ~np.isnan(Y[:, j])
            m = LogisticRegression(C=1e6, max_iter=1000).fit(logits[k, j, None], Y[k, j].astype(int))
            platt[f] = {"coef": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0])}
            probs[f] = m.predict_proba(logits[:, j, None])[:, 1]
        (outdir / "platt.json").write_text(json.dumps(platt, indent=2))
        out = pd.DataFrame({"study_id": t.study_id, "split": t.split, **{f"p_{f}": probs[f] for f in F},
                            **{f"logit_{f}": logits[:, j] for j, f in enumerate(F)}})
        out.to_parquet(outdir / "predictions_all_splits.parquet", index=False)
        out[out.split != "test"].to_parquet(outdir / "predictions_non_test.parquet", index=False)
        run.log(f"predictions written for all {len(out):,} studies (test {int((out.split == 'test').sum()):,}: predictions only, no metric)")
        rows = []
        for j, f in enumerate(F):
            for sp in ("calib", "thresh", "val"):
                msk = split == sp
                rows.append({"finding": f, "split": sp, **binary_metrics(Y[msk, j], probs[f][msk])})
        met = pd.DataFrame(rows)
        met.to_csv(outdir / "metrics_all_eval_splits.csv", index=False)
        val = met[met.split == "val"].set_index("finding")
        val.to_csv(outdir / "val_metrics.csv")
        vm = split == "val"
        viol = {e["id"]: int((probs[e["child"]][vm] > probs[e["parent"]][vm]).sum()) for e in kg.load_findings()["is_a"]}
        run.log(f"[{label}, epoch {epoch}] VAL per finding:\n" + val[["n", "pos", "auroc", "auprc", "ece"]].round(4).to_string())
        run.log(f"[{label}, epoch {epoch}] VAL (macro over {len(val)} outputs): auroc {val.auroc.mean():.4f}, auprc {val.auprc.mean():.4f}, ece {val.ece.mean():.4f}")
        run.log(f"val hierarchy violations (P(child) > P(parent)), of {int(vm.sum())} studies: {viol} -> total {sum(viol.values())}")
        run.metric(summary="val_macro", which=label, auroc=float(val.auroc.mean()), auprc=float(val.auprc.mean()), ece=float(val.ece.mean()),
                   hierarchy_violations=int(sum(viol.values())), epoch=epoch)


if __name__ == "__main__":
    main()
