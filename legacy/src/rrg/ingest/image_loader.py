"""N01: decode a chest X-ray and attach its metadata.

Scope (v1): the input is MIMIC-CXR-JPG (8-bit JPEG/PNG), so there is no DICOM decoding, no
VOI-LUT and no MONOCHROME1 handling here -- docs/pipeline.md 4.1 lists DICOM *or* pre-decoded
PNG/JPEG. A 16-bit or otherwise unsupported image is rejected rather than guessed at.
Model-specific preprocessing (resize / normalise) is the encoder's job (N03), because it comes
from the model's own config; the spatial branch (N02) is paused so there is only one branch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.config import Settings
from ..core.errors import InputDecodeError, InputModalityError
from ..core.types import DecodedImage, StudyMeta
from ..datasets.mimic_cxr import StudyRecord

_OK_MODES = {"L", "RGB"}


def meta_from_record(record: StudyRecord, settings: Settings) -> StudyMeta:
    return StudyMeta(
        study_uid=str(record.study_id),
        image_id=record.dicom_id,
        subject_id=str(record.subject_id),
        split=record.split,
        view_position=record.view_position,
        patient_orientation=record.patient_orientation,
        spatial_unavailable=not settings.spatial_enabled,
    )


def load_image(
    path: str | Path,
    settings: Settings,
    record: Optional[StudyRecord] = None,
) -> tuple[DecodedImage, StudyMeta]:
    """Decode `path`. With a `record`, metadata comes from MIMIC and the view is validated;
    without one, only what the file itself provides is recorded."""
    from PIL import Image, UnidentifiedImageError

    path = Path(path)
    if path.suffix.lower() not in {e.lower() for e in settings.input.accepted_extensions}:
        raise InputDecodeError(f"{path}: extension {path.suffix!r} not in input.accepted_extensions")

    if record is not None and record.view_position not in set(settings.dataset.accepted_view_positions):
        raise InputModalityError(
            f"{path}: view {record.view_position!r} is not in dataset.accepted_view_positions "
            f"{settings.dataset.accepted_view_positions}"
        )

    try:
        img = Image.open(path)
        img.load()  # force the decode now so a truncated file fails here, not later
    except FileNotFoundError as exc:
        raise InputDecodeError(f"{path}: file not found") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InputDecodeError(f"{path}: cannot decode image ({exc})") from exc

    if img.mode not in _OK_MODES:
        raise InputDecodeError(
            f"{path}: unsupported pixel mode {img.mode!r} (only 8-bit {sorted(_OK_MODES)} are accepted; "
            "convert 16-bit / palette images explicitly upstream)"
        )
    w, h = img.size
    if min(w, h) < settings.input.min_image_side:
        raise InputDecodeError(f"{path}: image {w}x{h} is smaller than input.min_image_side={settings.input.min_image_side}")

    if record is not None:
        meta = meta_from_record(record, settings)
    else:
        meta = StudyMeta(study_uid=path.parent.name, image_id=path.stem, spatial_unavailable=not settings.spatial_enabled)
    return DecodedImage(image=img, source_path=path, width=w, height=h, mode=img.mode), meta


def load_record_image(record: StudyRecord, settings: Settings) -> tuple[DecodedImage, StudyMeta]:
    return load_image(record.image_path, settings, record)
