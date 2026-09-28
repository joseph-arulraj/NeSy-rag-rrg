"""N17: the CBM head -- a single linear layer over the gathered concept scores, producing
per-label logits (pipeline.md §4.7: "Typically a single linear layer... so that weights are
directly inspectable"). Sigmoid is applied by the caller (train.py / infer.py), not baked in
here, so raw logits stay available for the required contribution attribution
(`w_p,i * x_i`, pipeline.md §4.7)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CBMHeadInfo:
    n_concepts: int
    n_labels: int
    label_space: list[str]
    bank_version_concepts_sha256: Optional[str]   # recorded when available, for version-mismatch detection


def build_head(n_concepts: int, n_labels: int):
    import torch.nn as nn
    return nn.Linear(n_concepts, n_labels)


def save_head(path: Path, head, info: CBMHeadInfo) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": head.state_dict(), "info": asdict(info)}, tmp)
    tmp.replace(path)


def load_head(path: Path):
    import torch
    import torch.nn as nn

    payload = torch.load(path, map_location="cpu", weights_only=False)
    info = CBMHeadInfo(**payload["info"])
    head = nn.Linear(info.n_concepts, info.n_labels)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, info
