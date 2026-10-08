"""Tests for the 2026-10-07 runner / evaluation additions (validate data only)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
PY = "/scratch/users/k23031260/.conda/envs/rrg/bin/python"


def test_label_rules_reproduce_labels_v2_on_val():
    from nesy import kg, labels as LB
    F = kg.finding_ids()
    t = pq.read_table(V2 / "data/labels/labels_v2.parquet", filters=[("split", "=", "val")]).to_pandas().set_index("study_id")
    src = t[[f"src_{f}" for f in F]].rename(columns=lambda c: c[4:])
    src["any_abnormality"] = 9                       # the root has no source column: starts blank
    got = LB.from_codes(src)
    np.testing.assert_array_equal(np.nan_to_num(got[F].values, nan=-5), np.nan_to_num(t[F].values, nan=-5))


def _parse(text, ents):
    """Minimal RadGraph-shaped parse: ents = [(tokens, label, start_ix, [linked entity keys])]."""
    e = {str(i): {"tokens": t, "label": l, "start_ix": s, "end_ix": s, "relations": [["located_at", str(r)] for r in rel]}
         for i, (t, l, s, rel) in enumerate(ents)}
    return {"text": text, "entities": e}


def test_comparator_v2_reads_dropped_sentences_and_keeps_v1():
    from nesy import rg_match as M
    p = _parse("Possible atelectasis . Possible support device . No focal consolidation .",
               [("atelectasis", "Observation::uncertain", 1, [])])
    exp = {"atelectasis": {"cert": "possible", "side": None, "zone": None},
           "support_devices": {"cert": "possible", "side": None, "zone": None},
           "consolidation": {"cert": "absent", "side": None, "zone": None}}
    assert M.compare(exp, M.claims(p, version=1), p["text"])          # v1: both sentences missed
    assert not M.compare(exp, M.claims(p, version=2), p["text"])      # v2: read lexically


def test_comparator_v2_still_catches_a_wrong_sentence():
    from nesy import rg_match as M
    p = _parse("Possible atelectasis . No support devices .", [("atelectasis", "Observation::uncertain", 1, [])])
    exp = {"atelectasis": {"cert": "possible", "side": None, "zone": None},
           "support_devices": {"cert": "possible", "side": None, "zone": None}}
    mm = M.compare(exp, M.claims(p, version=2), p["text"])
    assert any(m["kind"] == "polarity" and m["finding"] == "support_devices" for m in mm)


def test_comparator_v2_list_sentence_side_not_spread():
    from nesy import rg_match as M
    p = _parse("There is edema . Edema , and bilateral pleural effusion .",
               [("edema", "Observation::definitely present", 2, []), ("Edema", "Observation::definitely present", 4, []),
                ("bilateral", "Anatomy::definitely present", 7, []), ("pleural", "Anatomy::definitely present", 8, []),
                ("effusion", "Observation::definitely present", 9, [1, 2])])
    p["entities"]["4"]["relations"] = [["located_at", "3"], ["located_at", "2"]]
    exp = {"edema": {"cert": "present", "side": None, "zone": None},
           "pleural_effusion": {"cert": "present", "side": "bilateral", "zone": None}}
    assert any(m["kind"] == "side" and m["finding"] == "edema" for m in M.compare(exp, M.claims(p, version=1), p["text"]))
    assert not M.compare(exp, M.claims(p, version=2), p["text"])


@pytest.mark.parametrize("args,msg", [(["--split", "test"], "REFUSED"), (["--dataset", "vindr_test"], "REFUSED"),
                                      (["--dataset", "padchest_gr"], "REFUSED")])
def test_runner_refuses_test_and_external(args, msg):
    r = subprocess.run([PY, str(V2 / "scripts/run_pipeline.py"), *args, "--no-stage8"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and msg in (r.stderr + r.stdout)
    assert not any("pipeline" in p.name for p in (V2 / "runs").glob("*") if p.stat().st_mtime > __import__("time").time() - 5)


def test_evaluate_refuses_test():
    r = subprocess.run([PY, str(V2 / "scripts/evaluate.py"), "--pipeline-run", "x", "--split", "test"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "REFUSED" in (r.stderr + r.stdout)
