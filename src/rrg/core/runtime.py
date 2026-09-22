"""Device selection and process-level runtime setup (torch is imported lazily)."""
from __future__ import annotations

import os
import random

from .config import Settings
from .errors import ConfigError


def resolve_device(name: str):
    """'auto' -> cuda > mps > cpu. An explicitly requested device that is unavailable raises
    instead of silently falling back (a silent CPU fallback on the H100 node would be a bug)."""
    import torch

    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ConfigError("runtime.device=cuda but CUDA is not available in this environment")
    if name == "mps" and not mps_ok:
        raise ConfigError("runtime.device=mps but MPS is not available in this environment")
    return torch.device(name)


def configure_runtime(settings: Settings):
    """Seed, set threads / torch-hub cache, and return the resolved torch.device."""
    import numpy as np
    import torch

    rt = settings.runtime
    if rt.mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if rt.torch_hub_dir is not None:
        torch.hub.set_dir(str(rt.torch_hub_dir))
    if rt.torch_num_threads is not None:
        torch.set_num_threads(rt.torch_num_threads)
    random.seed(rt.seed)
    np.random.seed(rt.seed)
    torch.manual_seed(rt.seed)
    return resolve_device(rt.device)
