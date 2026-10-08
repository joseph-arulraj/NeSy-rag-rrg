"""N03 / N05: CLEAR image encoder -> the 768-d unit-norm feature X.

X is the single image representation used downstream: X . concept_bank -> concept scores (N07)
and X -> FAISS (N09). This is CLEAR's normalised `encode_image` output; CLEAR's separate
"LLM-projected" embedding is not used. Precision is fp32 only (config-enforced): the index and
every query embedding must be computed identically.

CLEAR is used only through `load_pretrained` (official loader + preprocessing) and
`model.encode_image`; nothing here depends on CLEAR's pipeline classes or private methods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from ..core.config import Settings, require_file
from ..core.errors import ClearLoadError, ClearNaNError
from ..core.hashing import file_sha256
from ..core.runtime import resolve_device
from ..core.types import DecodedImage, ImageEmbedding

if TYPE_CHECKING:
    import torch


class ClearEncoder:
    def __init__(self, settings: Settings, device: Optional["torch.device"] = None):
        self.settings = settings
        self.device = device or resolve_device(settings.runtime.device)
        cfg = settings.clear

        ckpt = require_file(settings.paths.clear_checkpoint, "paths.clear_checkpoint")
        try:
            import clear  # the CLEAR repo: pip install -e ./CLEAR
        except ImportError as exc:
            raise ClearLoadError(
                "cannot import 'clear'. Install the cloned CLEAR repo into this environment: pip install -e ./CLEAR"
            ) from exc
        try:
            self.model, self.preprocess = clear.load_pretrained(
                checkpoint_path=ckpt,
                device=self.device,
                local_files_only=cfg.local_files_only,
            )
        except Exception as exc:  # noqa: BLE001 -- wrap any loader failure with actionable context
            hint = ""
            if cfg.local_files_only:
                hint = " (local_files_only=true needs the DINOv2 source in the torch hub cache: run scripts/cache_dinov2.py)"
            raise ClearLoadError(f"CLEAR failed to load {ckpt}: {exc}{hint}") from exc
        self.model.eval()

        out_dim = getattr(self.model.visual, "output_dim", None)
        if out_dim is not None and int(out_dim) != cfg.embed_dim:
            raise ClearLoadError(f"CLEAR image dim {out_dim} != clear.embed_dim {cfg.embed_dim}")
        self.checkpoint_path = ckpt
        self.checkpoint_name = ckpt.name
        self._sha: Optional[str] = file_sha256(ckpt) if cfg.hash_checkpoint else None

    @property
    def checkpoint_sha256(self) -> str:
        """sha256 of the checkpoint (computed on first use if clear.hash_checkpoint is false)."""
        if self._sha is None:
            self._sha = file_sha256(self.checkpoint_path)
        return self._sha

    # ------------------------------------------------------------------ encoding
    def encode_tensors(self, batch: "torch.Tensor") -> "torch.Tensor":
        """Already-preprocessed [B, 3, H, W] tensors -> [B, D] float32 unit-norm features (CPU).
        Used directly by DataLoader-based bulk encoding, where workers do the preprocessing."""
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            feats = self.model.encode_image(batch.to(self.device)).float()
            if not torch.isfinite(feats).all():
                raise ClearNaNError("CLEAR produced NaN/Inf image features; refusing to propagate them")
            return F.normalize(feats, dim=-1).cpu()

    def encode_batch(self, images: Sequence[Union[DecodedImage, Any]]) -> "torch.Tensor":
        """PIL images (or DecodedImage) -> [B, D] float32 unit-norm features on CPU."""
        import torch

        pil = [im.image if isinstance(im, DecodedImage) else im for im in images]
        outputs = []
        bs = self.settings.clear.batch_size
        for start in range(0, len(pil), bs):
            batch = torch.stack([self.preprocess(im) for im in pil[start : start + bs]])
            outputs.append(self.encode_tensors(batch))
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.settings.clear.embed_dim))

    def encode(self, image: Union[DecodedImage, Any]) -> ImageEmbedding:
        vec = self.encode_batch([image])[0].numpy().astype(np.float32, copy=False)
        return ImageEmbedding(
            vector=vec,
            embed_dim=int(vec.shape[0]),
            checkpoint=self.checkpoint_name,
            checkpoint_sha256=self._sha,
            normalised=True,
            precision="fp32",
        )
