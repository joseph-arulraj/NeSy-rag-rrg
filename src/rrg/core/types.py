"""Shared data contracts. Only the types needed by the modules built so far;
the rest of docs/architecture.md section 3.1 is added as each module lands."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    MIDLINE = "midline"
    UNSPECIFIED = "unspecified"


class Polarity(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


AnatomyID = str    # a RadLex RID, e.g. "RID1327"
ConceptID = int    # index into the 368k concept bank, 0..368293


@dataclass(frozen=True)
class StudyMeta:
    """Per-image metadata (docs/pipeline.md 4.1). MIMIC-CXR-JPG has no DICOM header, so
    fields DICOM would supply (pixel spacing, acquisition time) are None."""

    study_uid: str                            # MIMIC study_id
    image_id: str                             # MIMIC dicom_id
    subject_id: Optional[str] = None
    split: Optional[str] = None               # official MIMIC-CXR split
    view_position: Optional[str] = None       # PA | AP | LATERAL | ...
    # MIMIC's PatientOrientationCodeSequence is posture (Erect/Recumbent), NOT left/right
    # orientation. Kept under the architecture's field name; do not use it to derive laterality.
    patient_orientation: Optional[str] = None
    pixel_spacing: Optional[tuple[float, float]] = None
    acquisition_datetime: Optional[str] = None
    is_inverted: bool = False
    spatial_unavailable: bool = True          # N02 paused in v1 -> Evidence A absent by design
    laterality_unreliable: bool = False


@dataclass(frozen=True)
class DecodedImage:
    """N01 output: a decoded 8-bit image plus provenance. Model-specific preprocessing
    (resize / normalise) belongs to the encoder (N03), not to N01."""

    image: Any                                # PIL.Image.Image (kept untyped so N01 needs no PIL import here)
    source_path: Path
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class ImageEmbedding:
    """N05: the 768-d L2-normalised CLEAR image feature 'X' that feeds BOTH the concept
    similarity (N07) and FAISS retrieval (N09)."""

    vector: np.ndarray                        # float32 [D], CPU
    embed_dim: int
    checkpoint: str
    checkpoint_sha256: Optional[str] = None
    normalised: bool = True
    precision: str = "fp32"                   # the FAISS branch is fp32-only; recorded so a mismatch is detectable


@dataclass(frozen=True)
class ConceptHit:
    concept_id: int
    text: str
    score: float


# ================================================================================================
# pipeline.md §3.2: Stage 2/3 shared contracts (belief graph). EvidenceSourceID is a plain str
# here (not an Enum) since it needs to hold "B"/"C"/"D" today and "A" once N02 exists, without a
# schema change; validity is enforced where these are constructed, not by the type itself.
# ================================================================================================
@dataclass
class EvidenceRef:
    source: str                       # "A" | "B" | "C" | "D"
    raw_score: float                  # source-native scale, pre-normalisation
    norm_score: float                 # [0,1] after fusion's calibration step
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    finding_id: str                   # stable within a study
    label: str                        # canonical finding term
    anatomy: Optional[AnatomyID]
    laterality: Laterality
    attributes: dict[str, str] = field(default_factory=dict)
    polarity: str = "uncertain"       # "present" | "absent" | "uncertain"
    confidence: float = 0.0           # [0,1], calibrated post-fusion
    support: list[EvidenceRef] = field(default_factory=list)


@dataclass
class Relation:
    kind: str                         # "located_at" | "modifies" | "suggests" | "contradicts"
    source_finding_id: str
    target_finding_id: str


@dataclass
class RuleApplication:
    """N25's audit record (pipeline.md §5.4): every removal/downweight must emit one of these."""
    rule_id: str
    rule_version: str
    targets: list[str]                # finding_id(s) the rule acted on
    action: str                       # "drop" | "downweight" | "resolve" | "collapse" | "drop_attribute"
    before: dict[str, Any]
    after: dict[str, Any]
    rationale: str


@dataclass
class BeliefGraph:
    findings: list[Finding] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    study_meta: Optional[StudyMeta] = None
    graph_version: str = "initial"    # "initial" | "verified"
    audit: list[RuleApplication] = field(default_factory=list)
    rejected: list[Finding] = field(default_factory=list)     # pipeline.md §5.3: excluded-but-retained-for-audit
    suppressed: list[Finding] = field(default_factory=list)   # pipeline.md §5.4 N25: non-destructive removal (INV-2/§5.4)
