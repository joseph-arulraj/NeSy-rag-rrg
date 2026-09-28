"""Exception hierarchy. One class per error code named in docs/pipeline.md."""
from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error the pipeline raises deliberately."""

    code = "E_PIPELINE"


class ConfigError(PipelineError):
    code = "E_CONFIG"


class InputDecodeError(PipelineError):
    code = "E_INPUT_DECODE"


class InputModalityError(PipelineError):
    """Input is decodable but is not an accepted chest X-ray view."""

    code = "E_INPUT_MODALITY"


class ClearLoadError(PipelineError):
    code = "E_CLEAR_LOAD"


class ClearNaNError(PipelineError):
    code = "E_CLEAR_NAN"


class BankLoadError(PipelineError):
    """Concept vocabulary / embedding files are missing, malformed or misaligned."""

    code = "E_BANK_LOAD"


class BankDimMismatchError(PipelineError):
    code = "E_BANK_DIM_MISMATCH"


class NonFiniteScoreError(PipelineError):
    code = "E_SCORE_NONFINITE"


class EmbeddingContractError(PipelineError):
    """An embedding violates the interface contract (wrong rank / not unit-norm)."""

    code = "E_EMBEDDING_CONTRACT"


class SplitLeakageError(PipelineError):
    """The data violates the split protocol (e.g. a patient appears in two splits)."""

    code = "E_SPLIT_LEAKAGE"


class IndexBuildError(PipelineError):
    code = "E_INDEX_BUILD"


class IndexVersionMismatchError(PipelineError):
    """The FAISS index was not built with the encoder / precision / dimension now in use."""

    code = "E_INDEX_VERSION"


class LLMUnavailableError(PipelineError):
    """N27: the LLM API could not be reached, or every retry was exhausted (pipeline.md §6.1)."""

    code = "E_LLM_UNAVAILABLE"


class LLMMalformedOutputError(PipelineError):
    """N27: the LLM returned an empty or structurally invalid response (pipeline.md §6.1). Not
    retried -- v1 has no regeneration path to recover into (pipeline.md rev 1)."""

    code = "E_LLM_MALFORMED"
