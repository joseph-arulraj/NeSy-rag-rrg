"""Item 11: load a random sample of VinDr-CXR and PadChest-GR images through the external loaders and
save contact sheets for visual checking of orientation, inversion and windowing. Loading only; no
labels are read and no metric is computed. MIMIC sheet included as the visual reference.

Outputs (run dir): contact_vindr_train.png, contact_vindr_test.png, contact_padchest_gr.png,
contact_mimic_reference.png, load_log.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nesy import external as X, manifests as M  # noqa: E402
from nesy.runlog import Run  # noqa: E402

TILE = 256


def sheet(images, labels, cols=6):
    rows = (len(images) + cols - 1) // cols
    out = Image.new("L", (cols * TILE, rows * (TILE + 14)), 0)
    d = ImageDraw.Draw(out)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        out.paste(im.resize((TILE, TILE)), (c * TILE, r * (TILE + 14) + 14))
        d.text((c * TILE + 2, r * (TILE + 14)), lab[:40], fill=255)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("external-contact-sheets", {**vars(args), "low_pct": X.LOW_PCT, "high_pct": X.HIGH_PCT}, args.run_dir) as run:
        log = []
        v = M.load("vindr", columns=["image_id", "image_path", "role"])
        for role in ["vindr_train", "external_test"]:
            s = v[v.split == role].sample(args.n, random_state=0)
            ims, labs = [], []
            for r in s.itertuples():
                im, info = X.load_vindr(r.image_path)
                ims.append(im)
                labs.append(f"{r.image_id[:10]} {info['photometric'][-1]}")
                log.append({"dataset": "vindr", "role": role, "id": r.image_id, **{k: str(val) for k, val in info.items()}})
            name = "contact_vindr_train.png" if role == "vindr_train" else "contact_vindr_test.png"
            sheet(ims, labs).save(run.dir / name)
            run.log(f"VinDr {role}: {len(ims)} images loaded -> {name}")
        p = M.load("padchest_gr", columns=["image_id", "archive_prefix", "archive_member", "role"]).sample(args.n, random_state=0)
        ims, labs = [], []
        for r in p.itertuples():
            im, info = X.load_padchest(r.archive_prefix, r.archive_member)
            ims.append(im)
            labs.append(r.image_id[:12])
            log.append({"dataset": "padchest_gr", "role": "external_test", "id": r.image_id, **{k: str(val) for k, val in info.items()}})
        sheet(ims, labs).save(run.dir / "contact_padchest_gr.png")
        run.log(f"PadChest-GR: {len(ims)} images loaded -> contact_padchest_gr.png")
        m = M.load("mimic", columns=["dicom_id", "image_path", "view", "is_frontal"])
        m = m[m.is_frontal & (m.split == "train")].sample(args.n, random_state=0)
        sheet([Image.open(x).convert("L") for x in m.image_path], [f"{d[:10]} {vw}" for d, vw in zip(m.dicom_id, m.view)]).save(
            run.dir / "contact_mimic_reference.png")
        run.log("MIMIC reference sheet written")
        pd.DataFrame(log).to_csv(run.dir / "load_log.csv", index=False)


if __name__ == "__main__":
    main()
