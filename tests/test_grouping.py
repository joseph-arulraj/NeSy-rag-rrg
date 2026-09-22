import numpy as np
import pytest

from rrg.concepts.bank import BankInfo, ConceptBank, ConceptTag, group_key_for
from rrg.concepts.grouping import group_and_rank
from rrg.core.config import load_settings
from rrg.core.errors import BankDimMismatchError
from rrg.core.types import Laterality, Polarity


def make_bank(texts, tags, scores):
    info = BankInfo("c", "e", len(texts), 8, "float32", 1.0, 1.0, True, True)
    bank = ConceptBank(tuple(texts), np.zeros((len(texts), 8), dtype="float32"), info)
    return bank.with_tags(tags), np.asarray(scores, dtype="float32")


def tag(finding, anatomy, lat, pol, temp="stationary"):
    return ConceptTag(anatomy=anatomy, laterality=lat, resolved_polarity=pol, temporal_class=temp,
                       canonical_finding=finding, group_key=group_key_for(finding, anatomy, lat))


@pytest.fixture
def cfg():
    return load_settings(overrides=["grouping.top_k_groups=10", "grouping.contributing_concepts_per_group=3",
                                     "grouping.present_absent_count_threshold=0.3"]).grouping


def test_near_duplicate_phrasings_collapse_into_one_group(cfg):
    """The measured problem this design fixes (pipeline.md 10.2): 4 near-duplicate device
    phrasings must become ONE group, not 4 top-K slots."""
    texts = [f"right basal pleural tube variant {i}" for i in range(4)] + ["small pleural effusion"]
    tags = [tag("support_devices", "RID_pleura", Laterality.RIGHT, Polarity.PRESENT)] * 4 + \
           [tag("pleural_effusion", "RID_pleura", Laterality.UNSPECIFIED, Polarity.PRESENT)]
    bank, scores = make_bank(texts, tags, [0.75, 0.72, 0.71, 0.70, 0.50])
    groups = group_and_rank(scores, bank, cfg)
    assert len(groups) == 2
    devices = next(g for g in groups if g.finding == "support_devices")
    assert devices.present_score == pytest.approx(0.75) and devices.present_count == 4
    assert len(devices.top_contributing_concepts) == 3   # bounded, not all 4


def test_present_and_absent_are_never_netted(cfg):
    """The bug this design specifically avoids: a high-scoring ABSENT concept must not be
    treated as strong evidence FOR the finding via a blind max() across polarities."""
    texts = ["pneumothorax", "no pneumothorax is present", "trace pneumothorax possible"]
    tags = [tag("pneumothorax", "RID_lung", Laterality.UNSPECIFIED, Polarity.PRESENT),
            tag("pneumothorax", "RID_lung", Laterality.UNSPECIFIED, Polarity.ABSENT),
            tag("pneumothorax", "RID_lung", Laterality.UNSPECIFIED, Polarity.PRESENT)]
    bank, scores = make_bank(texts, tags, [0.30, 0.90, 0.20])   # the ABSENT concept scores highest
    groups = group_and_rank(scores, bank, cfg)
    g = groups[0]
    assert g.finding == "pneumothorax"
    assert g.absent_score == pytest.approx(0.90) and g.present_score == pytest.approx(0.30)
    # a confident negative ranks the group highly (visible), but is never folded into present_score
    assert g.present_score != pytest.approx(0.90)


def test_confident_negative_outranks_a_weak_positive(cfg):
    """max(present_score, absent_score) ranking: a confidently-asserted absence is as
    informative as a confident presence and must not be buried below a weak positive elsewhere."""
    texts = ["definitely no consolidation", "possible faint opacity"]
    tags = [tag("consolidation", "RID_lung", Laterality.UNSPECIFIED, Polarity.ABSENT),
            tag("lung_opacity", "RID_lung", Laterality.UNSPECIFIED, Polarity.PRESENT)]
    bank, scores = make_bank(texts, tags, [0.85, 0.20])
    groups = group_and_rank(scores, bank, cfg)
    assert groups[0].finding == "consolidation" and groups[0].absent_score == pytest.approx(0.85)


def test_uncertain_and_unmapped_concepts_are_excluded(cfg):
    texts = ["equivocal finding", "unmappable phrase with no anchor", "clear finding"]
    tags = [tag("edema", "RID_lung", Laterality.UNSPECIFIED, Polarity.UNCERTAIN),
            ConceptTag(anatomy=None, laterality=Laterality.UNSPECIFIED, resolved_polarity=Polarity.PRESENT,
                       temporal_class="stationary", canonical_finding=None, group_key=None),
            tag("edema", "RID_lung", Laterality.UNSPECIFIED, Polarity.PRESENT)]
    bank, scores = make_bank(texts, tags, [0.99, 0.99, 0.40])
    groups = group_and_rank(scores, bank, cfg)
    assert len(groups) == 1 and groups[0].present_score == pytest.approx(0.40)   # only the mapped, non-uncertain one counts


def test_indeterminate_temporal_concepts_are_excluded():
    """resolved_polarity=None (e.g. classify_temporal returned indeterminate) must never enter a group."""
    t = tag("effusion", "RID_pleura", Laterality.UNSPECIFIED, None)  # resolved_polarity forced None below
    t = ConceptTag(anatomy="RID_pleura", laterality=Laterality.UNSPECIFIED, resolved_polarity=None,
                   temporal_class="indeterminate", canonical_finding="effusion", group_key="effusion|RID_pleura|unspecified")
    bank, scores = make_bank(["compared to prior effusion"], [t], [0.99])
    cfg = load_settings().grouping
    assert group_and_rank(scores, bank, cfg) == []


def test_top_k_groups_is_respected(cfg):
    texts = [f"finding{i}" for i in range(15)]
    tags = [tag(f"f{i}", "RID_x", Laterality.UNSPECIFIED, Polarity.PRESENT) for i in range(15)]
    bank, scores = make_bank(texts, tags, [1.0 - i * 0.01 for i in range(15)])
    groups = group_and_rank(scores, bank, cfg)
    assert len(groups) == cfg.top_k_groups == 10
    assert groups[0].finding == "f0"   # highest score first


def test_dimension_mismatch_raises(cfg):
    bank, scores = make_bank(["a"], [tag("f", None, Laterality.UNSPECIFIED, Polarity.PRESENT)], [0.5])
    with pytest.raises(BankDimMismatchError):
        group_and_rank(np.zeros(5, dtype="float32"), bank, cfg)


def test_untagged_bank_raises():
    info = BankInfo("c", "e", 1, 8, "float32", 1.0, 1.0, True, True)
    bank = ConceptBank(("x",), np.zeros((1, 8), dtype="float32"), info)   # no .with_tags()
    with pytest.raises(ValueError, match="no tags attached"):
        group_and_rank(np.zeros(1, dtype="float32"), bank, load_settings().grouping)
