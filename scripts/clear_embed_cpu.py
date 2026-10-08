"""Item 3: CLEAR embeddings on CPU for the one-frontal-per-study images of official validate and
test, plus a consistency check of 100 random cached train embeddings, then one embedding store.

Store (data/embeddings/): clear_frontal_v1.f16 (float16 memmap [N, 768], L2-normalised fp32
before casting) + clear_frontal_v1_index.parquet (row, dicom_id, study_id, subject_id, source).
Rows = the cached official-train embeddings (old outputs/index, verified here) + new val/test.

  python -u scripts/clear_embed_cpu.py --threads 12 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import clear_model, manifests as M, paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

OUT = P.V2 / "data/embeddings"
OLD_INDEX = clear_model.OLD / "legacy/outputs/index"


class DS:
    def __init__(self, paths, preprocess):
        self.paths, self.pre = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        try:
            im = Image.open(self.paths[i])
            im.load()
            return i, self.pre(im), True
        except Exception:  # noqa: BLE001 -- reported, not raised
            return i, torch.zeros(3, *self.size), False


def embed(run, model, pre, paths, workers, bs=32, label="images"):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    dl = DataLoader(DS(paths, pre), batch_size=bs, num_workers=workers, shuffle=False)
    out = np.zeros((len(paths), 768), np.float32)
    ok = np.zeros(len(paths), bool)
    prog = Progress(run, len(paths), f"{label} embedded", every_s=60)
    done = 0
    with torch.inference_mode():
        for idx, x, good in dl:
            g = good.bool()
            if g.any():
                f = F.normalize(model.encode_image(x[g]).float(), dim=-1).numpy()
                out[idx[g].numpy()] = f
                ok[idx[g].numpy()] = True
            done += len(idx)
            prog.update(done)
    prog.update(done, force=True)
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-check", type=int, default=100)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("clear-embed-cpu", {**vars(args), "checkpoint": str(clear_model.CHECKPOINT)}, args.run_dir) as run:
        import torch
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        t = time.time()
        model, pre = clear_model.load("cpu")
        run.log(f"CLEAR loaded on CPU in {time.time() - t:.0f}s; preprocess: {pre}")
        DS.size = pre.transforms[1].size
        man = M.load("mimic", columns=["dicom_id", "subject_id", "study_id", "view", "source_split", "image_path"])

        # -- consistency check against cached train embeddings
        old_manifest = json.loads((OLD_INDEX / "train_manifest.json").read_text())
        if old_manifest["clear_checkpoint_sha256"] != clear_model.CHECKPOINT_SHA256:
            raise RuntimeError("cached index was built with a different checkpoint")
        ids = json.loads((OLD_INDEX / "train_ids.json").read_text())
        cached = np.load(OLD_INDEX / "train_embeddings.npy", mmap_mode="r")
        run.log(f"cached train embeddings: {cached.shape}, dtype {cached.dtype}; ids {len(ids['dicom_id']):,}")
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(ids["dicom_id"]), args.n_check, replace=False))
        mi = man.set_index("dicom_id")
        paths = [mi.loc[ids["dicom_id"][i], "image_path"] for i in pick]
        new, ok = embed(run, model, pre, paths, args.workers, label="check images")
        cos = (new * np.asarray(cached[pick])).sum(1)
        run.log(f"consistency on {args.n_check} cached train images: cosine min {cos.min():.6f}, median {np.median(cos):.6f}, "
                f"mean {cos.mean():.6f}; all > 0.999: {bool((cos > 0.999).all())}; decode failures {int((~ok).sum())}")
        run.metric(check="consistency", n=args.n_check, cos_min=float(cos.min()), cos_median=float(np.median(cos)),
                   passed=bool((cos > 0.999).all()))
        if not (cos > 0.999).all():
            raise RuntimeError("cached embeddings do not reproduce (cosine <= 0.999); investigate before using them")

        # -- policy check: cached ids == one-frontal-per-study of official train
        tr = clear_model.one_frontal_per_study(man[man.source_split == "train"])
        same = set(tr.dicom_id) == set(ids["dicom_id"])
        run.log(f"cached ids equal the one-frontal-per-study policy on official train: {same} ({len(tr):,} vs {len(ids['dicom_id']):,})")
        if not same:
            raise RuntimeError("cached train ids do not match the selection policy")

        # -- new: official validate + test
        vt = clear_model.one_frontal_per_study(man[man.source_split.isin(["validate", "test"])])
        run.log(f"to embed: {len(vt):,} studies (validate {int((vt.source_split == 'validate').sum()):,}, "
                f"test {int((vt.source_split == 'test').sum()):,})")
        OUT.mkdir(parents=True, exist_ok=True)
        ckpt = OUT / "valtest_fp32_checkpoint.npz"
        if ckpt.exists() and list(np.load(ckpt, allow_pickle=True)["dicom_id"]) == vt.dicom_id.tolist():
            z = np.load(ckpt, allow_pickle=True)
            emb, ok = z["emb"], z["ok"]
            run.log(f"resumed val/test embeddings from {ckpt}")
        else:
            emb, ok = embed(run, model, pre, vt.image_path.tolist(), args.workers, label="val/test images")
            np.savez(ckpt, emb=emb, ok=ok, dicom_id=np.array(vt.dicom_id.tolist(), dtype=object))
            run.log(f"checkpoint written: {ckpt}")
        run.log(f"decode failures: {int((~ok).sum())}")
        if (~ok).any():
            raise RuntimeError(f"{int((~ok).sum())} images failed to decode: {vt.dicom_id[~ok].tolist()[:5]}")

        # -- unified float16 store
        n_tr = cached.shape[0]
        N = n_tr + len(vt)
        store = np.lib.format.open_memmap(OUT / "clear_frontal_v1.npy", mode="w+", dtype=np.float16, shape=(N, 768))
        for s in range(0, n_tr, 20000):
            e = min(s + 20000, n_tr)
            store[s:e] = np.asarray(cached[s:e], dtype=np.float16)
        store[n_tr:] = emb.astype(np.float16)
        store.flush()
        idx = pd.concat([
            pd.DataFrame({"dicom_id": ids["dicom_id"], "study_id": ids["study_id"], "subject_id": ids["subject_id"],
                          "source": "cached_old_index_fp32"}),
            pd.DataFrame({"dicom_id": vt.dicom_id.values, "study_id": vt.study_id.values, "subject_id": vt.subject_id.values,
                          "source": "cpu_v2"}),
        ], ignore_index=True)
        idx["row"] = np.arange(N)
        idx.to_parquet(OUT / "clear_frontal_v1_index.parquet", index=False)
        err = float(np.abs(np.linalg.norm(np.asarray(store[:1000], np.float32), axis=1) - 1).max())
        run.log(f"wrote {OUT / 'clear_frontal_v1.npy'} float16 {store.shape} ({(OUT / 'clear_frontal_v1.npy').stat().st_size / 1e6:.0f} MB); "
                f"max |norm-1| after fp16 cast on 1000 rows: {err:.4f}")
        run.metric(n_rows=N, n_new=len(vt))


if __name__ == "__main__":
    main()
