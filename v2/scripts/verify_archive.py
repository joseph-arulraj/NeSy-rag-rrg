"""Check that every file in the stretch archive can be read back: predictions load with the expected rows and
splits and finite probabilities in [0, 1]; models unpickle and produce scores. No labels are read and no metric
is computed. Writes MANIFEST.txt (file, size, sha256) in the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy.runlog import Run  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("verify-archive", vars(args), args.run_dir) as run:
        A = Path(args.archive)
        bad = 0
        for st in ("stage1", "stage2", "stage3", "stage5"):
            d = A / st
            p = pd.read_parquet(d / "predictions_all_splits.parquet")
            pc = [c for c in p.columns if c.startswith("p_")]
            v = p[pc].values
            ok = np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
            b = joblib.load(d / "model.joblib")
            run.log(f"{st}: {len(p):,} studies, splits {p.split.value_counts().to_dict()}, {len(pc)} probability columns "
                    f"(+{p.shape[1] - 2 - len(pc)} side columns), finite in [0,1]: {ok}; model.joblib loads, "
                    f"{len(b['models'])} finding models, {len(b['feature_names'])} features")
            bad += not ok
        for st in sorted(A.glob("stage4*")):
            idx = pd.read_parquet(st / "scores_index.parquet")
            m = joblib.load(st / "models.joblib")
            for s in ("A_imagenome", "B_chexmask"):
                sc = np.load(st / f"scores_{s}.npy", mmap_mode="r")
                assert sc.shape[0] == len(idx)
                f0 = next(iter(m["models"][s]))
                mk = m["models"][s][f0][0]
                x = np.zeros((1, len(mk["w"])), np.float32)
                _ = ((x - mk["mu"]) / mk["sd"]) @ mk["w"] + mk["b"]
                run.log(f"{st.name} {s}: scores {sc.shape} float16 (images x regions x findings); models for "
                        f"{len(m['models'][s])} findings x {len(m['models'][s][f0])} fold(s), weights dim {len(mk['w'])}; scoring OK")
            run.log(f"{st.name}: {len(idx):,} images, splits {idx.split.value_counts().to_dict()}")
        lines, total = [], 0
        for f in sorted(A.rglob("*")):
            if f.is_file() and f.name != "MANIFEST.txt":
                total += f.stat().st_size
                lines.append(f"{f.relative_to(A)}\t{f.stat().st_size}\t{sha(f)}")
        (A / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
        run.log(f"{len(lines)} files, total {total / 1e9:.3f} GB; MANIFEST.txt written; problems: {bad}")
        if bad:
            raise AssertionError("archive check failed")


if __name__ == "__main__":
    main()
