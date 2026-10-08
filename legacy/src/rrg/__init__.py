"""Evidence-based radiology report generation (neuro-symbolic RAG) -- v1.

Built from docs/pipeline.md (source of truth) and docs/architecture.md.
"""
import os as _os

# Unsupported ops on Apple MPS fall back to CPU instead of raising. Harmless on CUDA/CPU.
# Must be set before torch dispatches its first MPS op, hence at package import.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
