"""Stage 8 diagnosis (2026-10-07). Validate only.
1. RadGraph parses (modern-radgraph-xl, no section headers) of: every template text whose round trip failed,
   the first LLM attempt of every fallback study, and a set of probe sentences -> parses.json, for reading
   why the comparator misses them.
2. LLM retry check on --n-llm fallback studies: resend attempt 1's conversation with the feedback (as Stage 8
   did, temperature 0.3) and record whether the reply text changes. Prompts carry only the graph's statements.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nesy.runlog import Run  # noqa: E402

PROBES = ["Possible support device.", "Support devices are in place.", "No focal consolidation.", "Heart size is normal.",
          "The cardiomediastinal silhouette may be enlarged.", "The cardiomediastinal silhouette is enlarged.",
          "The cardiomediastinal silhouette is within normal limits.", "The heart size is borderline.",
          "Possible support devices.", "A support device is possibly present.", "There may be a support device.",
          "Lines and tubes are in place.", "Possible enlargement of the cardiomediastinal silhouette.",
          "The cardiomediastinal silhouette is possibly enlarged.", "No consolidation.", "The heart is normal in size.",
          "Possible atelectasis. A small pneumothorax cannot be excluded. Possible support device. No focal consolidation. Heart size is normal."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/20261007-0925_lane-final/stage8_results/results.jsonl")
    ap.add_argument("--n-llm", type=int, default=20)
    ap.add_argument("--key-file", default=str(Path.home() / ".rrg_llm_key"))
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("stage8-diag", vars(a), a.run_dir) as run:
        import yaml
        from radgraph import RadGraph
        from stage8_phrase import LLM, feedback, split_sections
        from nesy import paths as P
        rs = [json.loads(l) for l in open(a.results)]
        texts = {}
        for r in rs:
            if r["template_mismatch"] and r["fallback"]:
                texts[f"tpl:{r['study_id']}"] = f"{r['final_text']['findings']} {r['final_text']['impression']}"
            if r["fallback"] and r["attempts"] and r["attempts"][0].get("text"):
                texts[f"llm1:{r['study_id']}"] = r["attempts"][0]["text"]
        for i, s in enumerate(PROBES):
            texts[f"probe:{i}"] = s
        keys = list(texts)
        rg = RadGraph(model_type="modern-radgraph-xl", batch_size=32, cuda=0)
        parsed = rg([texts[k] for k in keys])
        out = {k: {"text": texts[k], "parse": parsed[str(i)]} for i, k in enumerate(keys)}
        (run.dir / "parses.json").write_text(json.dumps(out, indent=1, default=str))
        run.log(f"parsed {len(keys)} texts ({sum(k.startswith('tpl') for k in keys)} template, "
                f"{sum(k.startswith('llm1') for k in keys)} LLM first attempts, {len(PROBES)} probes)")
        for i, s in enumerate(PROBES):
            ents = out[f"probe:{i}"]["parse"].get("entities", {})
            run.log(f"PROBE {s!r}: " + "; ".join(f"{e['tokens']}={e['label']}" for e in ents.values()))
        # 2. retry check
        cfg = yaml.safe_load((P.V2 / "configs/llm.yaml").read_text())["llm"]
        llm = LLM(a.key_file, cfg)
        fb = [r for r in rs if r["fallback"] and r["attempts"] and r["attempts"][0].get("raw")][: a.n_llm]
        rows = []
        for r in fb:
            first = r["attempts"][0]
            mm = [m for m in first["mismatch"] if m["kind"] not in ("malformed", "llm_error")]
            hist = [{"role": "assistant", "content": first["raw"]}, {"role": "user", "content": feedback(mm)}]
            again0 = llm(r["prompt"], None, 0.0)
            retry = llm(r["prompt"], hist, 0.3)
            s0 = split_sections(again0) if again0 else None
            s1 = split_sections(retry) if retry else None
            rows.append({"study_id": r["study_id"], "feedback": feedback(mm),
                         "first_again_identical": (" ".join(s0) if s0 else None) == first["text"],
                         "retry_identical_to_first": (" ".join(s1) if s1 else None) == first["text"],
                         "first": first["text"], "retry": " ".join(s1) if s1 else retry})
        (run.dir / "retry_check.json").write_text(json.dumps(rows, indent=1))
        n = len(rows)
        run.log(f"retry check on {n} fallback studies: first attempt reproduced at T=0 {sum(x['first_again_identical'] for x in rows)}/{n}; "
                f"retry WITH feedback identical to first {sum(x['retry_identical_to_first'] for x in rows)}/{n}; LLM calls {llm.calls}, status {llm.status}")
        for x in rows[:6]:
            run.log(f"  study {x['study_id']}: feedback: {x['feedback']}\n    first: {x['first']}\n    retry: {x['retry']}")


if __name__ == "__main__":
    main()
