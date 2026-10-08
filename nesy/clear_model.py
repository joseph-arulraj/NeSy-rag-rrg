"""CLEAR image encoder (frozen), loaded from the local checkpoint with the official preprocessing."""
from __future__ import annotations

from pathlib import Path

OLD = Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg")
CHECKPOINT = OLD / "model_weights/best_model.pt"
CHECKPOINT_SHA256 = "da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03"   # as recorded by the old index
TORCH_HUB = OLD / "model_weights/torch_hub"


def load(device: str = "cpu"):
    import clear
    import torch
    torch.hub.set_dir(str(TORCH_HUB))
    model, preprocess = clear.load_pretrained(checkpoint_path=CHECKPOINT, device=device, local_files_only=True)
    model.eval()
    return model, preprocess


def one_frontal_per_study(df):
    """PA before AP, ties by smallest dicom_id: the policy the cached train embeddings used."""
    rank = {"PA": 0, "AP": 1}
    d = df[df.view.isin(rank)].copy()
    d["_r"] = d.view.map(rank)
    return d.sort_values(["study_id", "_r", "dicom_id"]).drop_duplicates("study_id").drop(columns="_r")
