"""The scripts are the user-facing entry points, so they get a real end-to-end run too."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args, env=None):
    import os

    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT, env=e, timeout=600)


@pytest.mark.assets
def test_explore_concepts_report_and_grep(settings, need_assets, tmp_path):
    r = run("explore_concepts.py", "--set", f"paths.output_dir={tmp_path}")
    assert r.returncode == 0, r.stderr
    stats = json.loads((tmp_path / "concept_exploration.json").read_text())
    assert stats["n_concepts"] == 368294 and stats["duplicates_exact"] == 0
    assert (tmp_path / "concept_exploration.md").read_text().startswith("# CLEAR concept vocabulary")
    g = run("explore_concepts.py", "--grep", "^no pneumothorax", "--max", "3", "--no-write")
    assert g.returncode == 0 and "match /^no pneumothorax/" in g.stdout


@pytest.mark.assets
def test_run_demo_end_to_end(settings, need_assets, need_sample, dinov2_hub_dir, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("clear")
    r = run("run_demo.py", "--set", f"paths.output_dir={tmp_path}", "--set", f"runtime.torch_hub_dir={dinov2_hub_dir}",
            "--set", "clear.local_files_only=true", "--set", "demo.num_images=2", "--top-k", "5",
            env={"PYTHONPATH": str(ROOT / "CLEAR" / "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads((tmp_path / "demo_top_concepts.json").read_text())
    assert len(out) == 2 and all(len(o["top_concepts"]) == 5 for o in out)
    scores = [c["score"] for c in out[0]["top_concepts"]]
    assert scores == sorted(scores, reverse=True) and all(-1 <= x <= 1 for x in scores)


@pytest.mark.assets
def test_build_index_and_retrieval_demo_scripts(settings, need_assets, need_sample, dinov2_hub_dir, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("clear")
    common = ["--set", f"paths.index_dir={tmp_path / 'idx'}", "--set", f"paths.output_dir={tmp_path}",
              "--set", f"runtime.torch_hub_dir={dinov2_hub_dir}", "--set", "clear.local_files_only=true"]
    env = {"PYTHONPATH": str(ROOT / "CLEAR" / "src")}
    b = run("build_faiss_index.py", "--limit", "12", *common, env=env)
    assert b.returncode == 0, b.stderr[-2000:]
    assert (tmp_path / "idx" / "train_manifest.json").is_file() and '"built_limit": 12' in b.stdout
    r = run("run_retrieval_demo.py", "--query-split", "train", "--num", "2", *common, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "SMOKE INDEX" in r.stdout and "SAME IMAGE" not in r.stdout and "SAME PATIENT" not in r.stdout
    leak = run("run_retrieval_demo.py", "--query-split", "train", "--num", "1", "--no-exclude", *common, env=env)
    assert "SAME IMAGE" in leak.stdout
