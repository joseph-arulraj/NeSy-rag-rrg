"""One unit test per Stage 6/7 rule (KG schema, grounding G1-G6, decisions D1-D4, belief graph B1-B3,
presentation P1-P5). Run: python -m pytest tests -q"""
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import yaml  # noqa: E402

from nesy import belief as B, grounding as G, kg, report as R  # noqa: E402

THR = {f: {"present": 0.7, "possible": 0.4, "absent": 0.2} for f in kg.finding_ids()}


def probs(**over):
    p = {f: 0.05 for f in kg.finding_ids()}
    p.update(over)
    return p


# ---------------------------------------------------------------- KG schema validation
def _raw():
    return yaml.safe_load((kg.KG_DIR / "findings.yaml").read_text())


def _anat_ids():
    return set(kg.load_anatomy()["_ids"])


def test_K1_kg_files_validate():
    assert len(kg.finding_ids()) == 14          # 13 findings + any_abnormality root
    assert len(kg.load_anatomy()["_ids"]) >= 36


def test_K2_unknown_reference_rejected():
    d = _raw()
    d["is_a"].append({"id": "isa_x", "child": "atelectasis", "parent": "not_a_finding", "rationale": "x"})
    with pytest.raises(kg.KGError, match="unknown finding"):
        kg.validate_findings(d, _anat_ids())


def test_K3_is_a_cycle_rejected():
    d = _raw()
    d["is_a"].append({"id": "isa_x", "child": "lung_opacity", "parent": "pneumonia", "rationale": "x"})
    with pytest.raises(kg.KGError, match="cycle"):
        kg.validate_findings(d, _anat_ids())


def test_K4_missing_rationale_rejected():
    d = _raw()
    d["associated_with"][0].pop("rationale")
    with pytest.raises(kg.KGError, match="rationale"):
        kg.validate_findings(d, _anat_ids())


def test_K5_every_imagenome_region_maps_to_anatomy():
    regions = [l.strip().rstrip(",") for l in (Path("/scratch/prj/bhi_zihe_imaging/mimic_cxr_full/Chest_ImaGenome/semantics/"
                                                   "objects_detectable_by_bbox_pipeline_v1.txt").read_text().splitlines()) if l.strip()]
    assert len(regions) == 36
    assert all(G.region_to_anatomy(r) for r in regions), [r for r in regions if not G.region_to_anatomy(r)]


# ---------------------------------------------------------------- grounding
def test_G1_region_to_anatomy():
    assert G.region_to_anatomy("left lower lung zone") == "left_lower_lung_zone"
    assert G.region_to_anatomy("nonsense") is None


def test_G2_part_of_resolves_lung_side_zone_never_lobe():
    r = G.resolve_part_of("lingula")
    assert "lobe" not in r and r["lung"] == "left_lung" and r["side"] == "left" and r["zone"] is None
    z = G.resolve_part_of("right_lower_lung_zone")
    assert z["zone"] == "lower" and z["side"] == "right"
    assert G.zone_of("lung_left_upper") == "upper" and G.zone_of("right apical zone") == "upper"
    assert G.zone_of("lower lobe of right lung") is None


def test_G2_ground_output_has_no_lobe():
    out = G.ground("consolidation", [("lower lobe of left lung", 0.9), ("left lower lung zone", 0.8)])
    for loc in out["localisations"]:
        assert not hasattr(loc, "lobe")
    assert [l.zone for l in out["localisations"]] == [None, "lower"] and out["side"] == "left"

def test_G3_image_to_patient_flip():
    assert G.image_to_patient(0.8, "PA") == "left"       # image right = patient left
    assert G.image_to_patient(0.2, "AP") == "right"
    assert G.image_to_patient(0.8, "PA", horizontally_flipped=True) == "right"
    assert G.image_to_patient(0.51, "PA") == "midline"
    assert G.image_to_patient(0.8, "LATERAL") == "unknown"


def test_G4_may_occur_in_flags_without_deleting():
    out = G.ground("pneumothorax", [("right apical zone", 0.9), ("cardiac silhouette", 0.9)])
    assert len(out["localisations"]) == 2                  # nothing deleted
    v = {l.region: l.valid for l in out["localisations"]}
    assert v == {"right apical zone": True, "cardiac silhouette": False}


def test_G5_ctr_ap_flagged():
    assert G.ctr(0.55, "PA")["reliable"] is True
    assert G.ctr(0.55, "AP")["reliable"] is False
    assert G.ctr(float("nan"), "PA")["ctr"] is None


def test_G6_side_from_regions():
    assert G.ground("pleural_effusion", [("left costophrenic angle", 0.9)])["side"] == "left"
    assert G.ground("pleural_effusion", [("left costophrenic angle", 0.9), ("right costophrenic angle", 0.8)])["side"] == "bilateral"
    assert G.ground("pleural_effusion", [("left costophrenic angle", 0.3)])["side"] is None     # below min_score


# ---------------------------------------------------------------- decisions
def test_D1_four_bands():
    t = {"present": 0.7, "possible": 0.4, "absent": 0.2}
    assert [B.band(p, t) for p in (0.9, 0.5, 0.3, 0.1)] == ["present", "possible", "silent", "absent"]
    assert B.band(None, t) == "silent" and B.band(float("nan"), t) == "silent" and B.band(0.9, None) == "silent"


def test_D1_unreachable_tiers_never_entered():
    t = {"present": None, "possible": 0.4, "absent": None}      # e.g. fracture
    assert [B.band(p, t) for p in (0.99, 0.5, 0.01)] == ["possible", "possible", "silent"]


def test_D2_parent_raised_to_child_with_edge():
    g = B.build(1, probs(pneumonia=0.9, consolidation=0.1, lung_opacity=0.1), THR)
    assert g.nodes["consolidation"].band == "present" and g.nodes["lung_opacity"].band == "present"
    assert g.nodes["any_abnormality"].band == "present"
    raised = [a for a in g.audit if a.rule == "D2"]
    assert {a.edge for a in raised} == {"isa_005", "isa_002", "isa_101"}


def test_D2_silent_child_does_not_change_parent():
    g = B.build(1, probs(fracture=0.3), THR)                      # silent child (not critical), absent root
    assert g.nodes["fracture"].band == "silent" and g.nodes["any_abnormality"].band == "absent"
    assert g.no_acute_abnormality is True


def test_D3_no_acute_abnormality_from_root_band():
    assert B.build(1, probs(), THR).no_acute_abnormality is True
    assert B.build(1, probs(support_devices=0.99), THR).no_acute_abnormality is True       # devices are not a child
    assert B.build(1, probs(edema=0.5), THR).no_acute_abnormality is False                  # possible child lifts root
    assert B.build(1, probs(any_abnormality=0.5), THR).no_acute_abnormality is False        # root itself not absent


def test_D4_empty_absent_band_fails():
    assert B.check_bands({"edema": [0.1, 0.5, 0.9]}, THR)["edema"]["absent"] == 1
    with pytest.raises(ValueError, match="D4"):
        B.check_bands({"edema": [0.5, 0.9]}, THR)
    assert B.check_bands({"edema": [0.5, 0.9]}, {"edema": {"present": 0.7, "possible": 0.4, "absent": None}})   # unreachable: ok


def test_D5_factorised_marginals_never_exceed_parent():
    import numpy as np
    rng = np.random.default_rng(0)
    cond = {f: rng.uniform(size=1000) for f in kg.finding_ids()}
    m = B.marginals(cond)
    for e in kg.load_findings()["is_a"]:
        assert (m[e["child"]] <= m[e["parent"]] + 1e-12).all(), e["id"]
    assert np.allclose(m["pneumonia"], cond["pneumonia"] * cond["consolidation"] * cond["lung_opacity"] * cond["any_abnormality"])


# ---------------------------------------------------------------- belief graph
def test_B1_one_node_per_finding():
    g = B.build(1, probs(), THR)
    assert set(g.nodes) == set(kg.finding_ids())
    with pytest.raises(ValueError, match="B1"):
        g.add(B.Node("edema", 0.1, "absent"))


def test_B2_edges_only_from_kg():
    g = B.build(1, probs(), THR)
    kgd = kg.load_findings()
    assert {e["id"] for e in g.edges} == {e["id"] for e in kgd["is_a"]} | {e["id"] for e in kgd["associated_with"]}


def test_B3_every_change_audited():
    g = B.build(1, probs(), THR)
    n0 = len(g.audit)
    g.set("edema", "band", "present", "TEST", "unit test")
    assert len(g.audit) == n0 + 1 and g.audit[-1].before == "absent" and g.audit[-1].rule == "TEST"
    g.set("edema", "band", "present", "TEST", "no-op")             # unchanged value -> no entry
    assert len(g.audit) == n0 + 1


# ---------------------------------------------------------------- presentation
def test_P1_parent_omitted_when_child_present():
    rep = R.render(B.build(1, probs(pneumonia=0.9), THR))
    stated = {s["finding"] for s in rep["sentences"]}
    assert "pneumonia" in stated and "consolidation" not in stated and "lung_opacity" not in stated
    assert any(o["rule"] == "P1" and o["finding"] == "lung_opacity" for o in rep["omitted"])


def test_P2_only_pertinent_negatives():
    rep = R.render(B.build(1, probs(), THR))
    negs = {s["finding"] for s in rep["sentences"] if s["band"] == "absent"}
    assert negs == {"pneumothorax", "pleural_effusion", "consolidation", "cardiomegaly"}
    rep2 = R.render(B.build(1, probs(cardiomegaly=0.9), THR))          # conditional pertinent negative
    assert "edema" in {s["finding"] for s in rep2["sentences"] if s["band"] == "absent"}


def test_P3_silent_not_mentioned():
    thr = copy.deepcopy(THR)
    thr["pleural_other"] = {"present": None, "possible": None, "absent": None}
    rep = R.render(B.build(1, probs(pleural_other=0.99), thr))
    assert "pleural_other" not in {s["finding"] for s in rep["sentences"]}
    assert any(o["rule"] == "P3" and o["finding"] == "pleural_other" for o in rep["omitted"])
    assert "any_abnormality" not in {s["finding"] for s in rep["sentences"]}          # root never stated


def test_P4_possible_hedged_with_side():
    gr = G.ground("pleural_effusion", [("left costophrenic angle", 0.9)])
    gr["side_conf"] = 0.95
    g = B.build(1, probs(pleural_effusion=0.5), THR, grounding={"pleural_effusion": gr})
    s = next(s for s in R.render(g)["sentences"] if s["finding"] == "pleural_effusion")
    assert s["text"] == "Possible small left pleural effusion." and s["rule"] == "P4"


def test_P7_no_acute_abnormality_impression():
    rep = R.render(B.build(1, probs(), THR))
    assert rep["impression"] == "No acute cardiopulmonary abnormality."
    rep2 = R.render(B.build(1, probs(any_abnormality=0.3), THR))                        # root silent
    assert rep2["impression"] == "No finding meets the reporting threshold."
    rep3 = R.render(B.build(1, probs(pneumonia=0.5), THR))
    assert rep3["impression"].startswith("Pneumonia is possible")                       # capitalised (bug fix)


def test_P5_always_returns_a_report():
    g = B.build(1, probs(), THR)
    g.nodes["edema"].band = None                       # corrupt graph
    g.nodes["edema"].finding = None
    rep = R.render(g)
    assert rep["findings"] and rep["impression"]


def test_label_closure_rules_match_kg():
    """Item 1 closure and Stage 6 D2 use the same is_a edges."""
    assert set(kg.ancestors("pneumonia")) == {"consolidation", "lung_opacity", "any_abnormality"}
    assert kg.topo_order().index("pneumonia") < kg.topo_order().index("consolidation") < kg.topo_order().index("lung_opacity")


def test_P6_side_only_when_confident_and_lateralisable():
    gr = G.ground("pleural_effusion", [("left costophrenic angle", 0.9)])
    gr["side_conf"] = 0.6                                           # below side_min_confidence
    s = next(s for s in R.render(B.build(1, probs(pleural_effusion=0.9), THR, grounding={"pleural_effusion": gr}))["sentences"]
             if s["finding"] == "pleural_effusion")
    assert s["text"] == "There is a pleural effusion." and s["side_withheld"]
    gr2 = G.ground("edema", [("left lower lung zone", 0.9)])
    gr2["side_conf"] = 0.99                                         # edema is not lateralisable
    g = B.build(1, probs(edema=0.9), THR, grounding={"edema": gr2})
    assert g.nodes["edema"].side is None
    assert kg.load_findings()["_by_id"]["edema"]["lateralisable"] is False


def test_D1_absent_has_priority_over_possible_for_common_findings():
    t = {"present": 0.8, "possible": 0.02, "absent": 0.2}         # e.g. any_abnormality, prevalence 0.6
    assert [B.band(p, t) for p in (0.9, 0.5, 0.1)] == ["present", "possible", "absent"]


def test_N1_normal_call_requires_critical_findings_absent():
    thr = copy.deepcopy(THR)
    A = kg.load_findings()["normal_call"]["candidates"]["A"]
    assert B.build(1, probs(), thr, critical=A).no_acute_abnormality is True
    g = B.build(1, probs(pneumothorax=0.3), thr, critical=A)          # root absent, critical pneumothorax silent
    assert g.nodes["any_abnormality"].band == "absent" and g.no_acute_abnormality is False
    assert "critical" in g.audit[-1].reason and g.audit[-1].edge == "n1_critical_absent"
    rep = R.render(g)
    assert rep["impression"] != "No acute cardiopulmonary abnormality."       # nothing said about normality
    g2 = B.build(1, probs(edema=0.3), thr, critical=A)                # edema silent: not on list A
    assert g2.no_acute_abnormality is True
    assert B.build(1, probs(edema=0.3), thr, critical=kg.load_findings()["normal_call"]["candidates"]["B"]).no_acute_abnormality is False


def test_N1_inactive_by_default_root_alone_decides():
    assert kg.load_findings()["normal_call"]["critical"] == []
    assert B.build(1, probs(pneumothorax=0.3), THR).no_acute_abnormality is True       # silent critical finding ignored


def test_G6_side_only_attached_to_stated_findings():
    gr = G.ground("pleural_effusion", [("left costophrenic angle", 0.9)])
    gr["side_conf"] = 0.95
    g = B.build(1, probs(pleural_effusion=0.05), THR, grounding={"pleural_effusion": gr})      # absent
    assert g.nodes["pleural_effusion"].side is None


def test_P8_zone_only_with_stated_side_and_never_lobe():
    from nesy import pipeline_graph as PG
    thr = {f: {"present": 0.7, "possible": 0.4, "absent": 0.05, "reportable": True} for f in kg.finding_ids()}
    row = {f"p_{f}": 0.01 for f in kg.finding_ids()}
    row.update({"p_any_abnormality": 0.95, "p_lung_opacity": 0.9, "p_consolidation": 0.9,
                "pside_consolidation_left": 0.05, "pside_consolidation_right": 0.9, "pside_consolidation_bilateral": 0.05})
    ev = {"consolidation": {"support": 2.0, "left": -1.0, "right": 2.0, "zone": {"right": ("lower", 2.0), "left": ("upper", -1.0)}}}
    g = PG.build(1, row, thr, ev, {"consolidation": 0.5})
    rep = R.render(g)
    s = [x for x in rep["sentences"] if x["finding"] == "consolidation"][0]
    assert s["side"] == "right" and s["zone"] == "lower" and "right" in s["text"] and "lower zone" in s["text"]
    assert "lobe" not in (rep["findings"] + rep["impression"]).lower()
    # below the region threshold: no zone, side still stated
    g2 = PG.build(2, row, thr, ev, {"consolidation": 3.0})
    s2 = [x for x in R.render(g2)["sentences"] if x["finding"] == "consolidation"][0]
    assert s2["zone"] is None and s2["side"] == "right"


def test_R1_demotes_unsupported_present_and_R2_withholds_disagreeing_side():
    from nesy import pipeline_graph as PG
    thr = {f: {"present": 0.7, "possible": 0.4, "absent": 0.05, "reportable": True} for f in kg.finding_ids()}
    row = {f"p_{f}": 0.01 for f in kg.finding_ids()}
    row.update({"p_any_abnormality": 0.95, "p_pleural_effusion": 0.9,
                "pside_pleural_effusion_left": 0.9, "pside_pleural_effusion_right": 0.05, "pside_pleural_effusion_bilateral": 0.05})
    ev = {"pleural_effusion": {"support": 0.1, "left": 0.1, "right": 0.0, "zone": {}}}
    g = PG.build(1, row, thr, ev, {"pleural_effusion": 0.5}, use_r1=True, use_r2=True)
    n = g.nodes["pleural_effusion"]
    assert n.band == "possible" and n.prob == 0.9 and n.side is None
    assert {a.rule for a in g.audit} >= {"R1", "R2"}


def test_C1_case_evidence_changes_nothing_and_line_is_optional():
    from nesy import case_evidence as CE, pipeline_graph as PG
    thr = {f: {"present": 0.7, "possible": 0.4, "absent": 0.05, "reportable": True} for f in kg.finding_ids()}
    row = {f"p_{f}": 0.01 for f in kg.finding_ids()}
    row.update({"p_any_abnormality": 0.95, "p_pleural_effusion": 0.9})
    g = PG.build(1, row, thr, {}, {})
    before = {f: (n.prob, n.band, n.side) for f, n in g.nodes.items()}
    off = R.render(g, show_cases=False)
    crow = {"nb_study_ids": [11, 12, 13, 14, 15], "nb_sims": [0.9] * 5,
            **{f"case_pos_{f}": (4 if f == "pleural_effusion" else 0) for f in kg.finding_ids()},
            **{f"case_known_{f}": 5 for f in kg.finding_ids()}}
    CE.attach(g, crow)
    assert {f: (n.prob, n.band, n.side) for f, n in g.nodes.items()} == before
    assert any(a.rule == "C1" and "no probability or band was changed" in a.reason for a in g.audit)
    assert R.render(g, show_cases=False)["findings"] == off["findings"]
    on = R.render(g, show_cases=True)
    assert "seen in 4 of 5 similar cases" in on["findings"]
    assert all("case_line" not in s or s["text"] == t["text"] for s, t in zip(on["sentences"], off["sentences"]))
    assert "report" not in str(g.cases).lower()


def test_radlex_map_and_devices_validate():
    r = kg.load_radlex()
    by = {(e["kind"], e["key"]): e for e in r["entries"]}
    assert all(("finding", f) in by for f in kg.finding_ids())
    assert by[("finding", "pleural_effusion")]["radlex_id"] == "RID34539" and by[("finding", "pleural_effusion")]["match"] == "exact"
    assert by[("finding", "cardiomegaly")]["match"] == "none" and by[("finding", "cardiomegaly")]["radlex_id"] is None
    assert all(e["review"] == "needed" for e in r["entries"] if e["match"] != "exact")
    assert len(kg.load_devices()["device_types"]) == 17


def test_radlex_validation_rejects_forced_or_unknown():
    import copy
    d = copy.deepcopy(kg.load_radlex())
    d["entries"][0]["match"] = "close"
    with pytest.raises(kg.KGError):
        kg.validate_radlex(d, kg.finding_ids(), [n["id"] for n in kg.load_anatomy()["nodes"]], [])
    d = copy.deepcopy(kg.load_radlex())
    d["entries"].append({"kind": "finding", "key": "no_such_finding", "match": "none", "radlex_id": None, "source": "x"})
    with pytest.raises(kg.KGError):
        kg.validate_radlex(d, kg.finding_ids(), [n["id"] for n in kg.load_anatomy()["nodes"]], [e["id"] for e in kg.load_devices()["device_types"]])
