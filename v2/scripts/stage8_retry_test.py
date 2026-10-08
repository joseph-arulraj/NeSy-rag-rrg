"""Stage 8 fix test (2026-10-07), validate only, on --n studies that FELL BACK in run 20261007-093237_stage8-car.
A. Phrase them with comparator v2 and retry feedback v2 (template sentence quoted), up to --max-attempts:
   first-pass match, rescued by a retry, fallback. Same prompt, model and temperature schedule as before.
B. Guard sensitivity (does v2 still catch wrong text?): for the same studies' template texts, three corruptions
   per study where applicable - drop one stated sentence, negate one stated sentence ("Possible X." -> "No X."),
   add a side to one sentence whose finding has no side - parsed with RadGraph and compared with v1 and v2.
   A corruption is caught if compare() returns any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "scripts"))

from nesy.runlog import Run  # noqa: E402


def corruptions(tpl: dict, rng) -> list[tuple[str, str]]:
    """-> [(kind, findings + impression text)] built from the template's sentences."""
    sents = tpl["sentences"]
    stated = [s for s in sents if s["band"] in ("present", "possible") and s["finding"] != "any_abnormality"]
    out = []
    base = [s["text"] for s in sents]
    imp = tpl["impression"]
    if stated:
        s = rng.choice(stated)
        out.append(("drop_statement", " ".join(t for t in base if t != s["text"]) + " " + " ".join(
            x for x in re.split(r"(?<=\.)\s+", imp) if x and x != s["text"])))
        lab = s["finding"].replace("_", " ")
        neg = {"lung_opacity": "No focal lung opacity.", "support_devices": "No support devices.",
               "enlarged_cardiomediastinum": "The cardiomediastinal silhouette is within normal limits.",
               "cardiomegaly": "Heart size is normal."}.get(s["finding"], f"No {lab}.")
        out.append(("negate_statement", " ".join(neg if t == s["text"] else t for t in base) + " " + imp.replace(s["text"], neg)))
        noside = [x for x in stated if not x.get("side") and x["finding"] in ("lung_opacity", "atelectasis", "consolidation",
                                                                             "pleural_effusion", "pneumothorax", "lung_lesion")]
        if noside:
            x = rng.choice(noside)
            t2 = re.sub(r"\b(lung opacity|atelectasis|consolidation|pleural effusion|pneumothorax|pulmonary nodule|lung nodule)\b",
                        r"left \1", x["text"], count=1)
            if t2 != x["text"]:
                out.append(("add_side", " ".join(t2 if t == x["text"] else t for t in base) + " " + imp.replace(x["text"], t2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="run_pipeline graphs.jsonl (val)")
    ap.add_argument("--old", default="runs/20261007-0925_lane-final/stage8_results/results.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--key-file", default=str(Path.home() / ".rrg_llm_key"))
    ap.add_argument("--n-repair", type=int, default=150)
    ap.add_argument("--only-core", action="store_true")
    ap.add_argument("--skip-a", action="store_true", help="skip part A (already measured)")
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("stage8-retry-test", vars(a), a.run_dir) as run:
        import yaml
        from radgraph import RadGraph
        from nesy import rg_match as M, stage8 as S8
        old = [json.loads(l) for l in open(a.old)]
        failed = [r["study_id"] for r in old if r["fallback"]]
        rng = random.Random(0)
        ids = set(rng.sample(failed, min(a.n, len(failed))))
        G = [json.loads(l) for l in open(a.graphs)]
        G = [g for g in G if g["study_id"] in ids]
        run.log(f"{len(failed)} fallback studies in the old run; testing {len(G)}")
        rg = RadGraph(model_type="modern-radgraph-xl", batch_size=32, cuda=0)
        cfg = yaml.safe_load((V2.parent / "configs/default.yaml").read_text())["llm"]
        llm = S8.LLM(a.key_file, cfg)
        items = [{"study_id": g["study_id"], "template": g["template"], "no_acute": g["graph"]["no_acute_abnormality"]} for g in G]
        res = []
        for c in (range(0, len(items), 32) if not a.skip_a else []):
            res += S8.phrase(items[c:c + 32], llm, rg, a.max_attempts, 2, "feedback", "v2")
        n = max(len(res), 1)
        A = {"n": n, "first_pass_match": sum(r["first_pass_match"] for r in res),
             "rescued_by_retry": sum((not r["fallback"]) and not r["first_pass_match"] for r in res),
             "fallback": sum(r["fallback"] for r in res), "template_roundtrip_fail": sum(bool(r["template_mismatch"]) for r in res),
             "retry_identical_to_previous": sum(1 for r in res for i in range(1, len(r["attempts"]))
                                                if r["attempts"][i].get("text") == r["attempts"][i - 1].get("text")),
             "retries": sum(max(len(r["attempts"]) - 1, 0) for r in res), "llm_calls": llm.calls, "llm_status": llm.status}
        run.log(f"A. v2 comparator + v2 feedback on {n} previously failed studies: {A}")
        for r in [r for r in res if not r["first_pass_match"]][:8]:
            run.log(f"  study {r['study_id']} fallback={r['fallback']}: " + " || ".join(
                f"{x.get('text')} -> {[(m['kind'], m.get('finding')) for m in x['mismatch']]}" for x in r["attempts"]))
        with open(run.dir / "phrase_results.jsonl", "w") as fh:
            for r in res:
                fh.write(json.dumps(r, default=str) + "\n")
        # B. guard sensitivity
        cor = []
        for g in G:
            for kind, txt in corruptions(g["template"], rng):
                cor.append((g, kind, " ".join(txt.split())))
        parsed = rg([t for _, _, t in cor])
        from collections import Counter
        caught = {1: Counter(), 2: Counter()}
        tot = Counter()
        for j, (g, kind, txt) in enumerate(cor):
            exp = M.expected(g["template"], g["graph"]["no_acute_abnormality"])
            tot[kind] += 1
            for v in (1, 2):
                caught[v][kind] += bool(M.compare(exp, M.claims(parsed[str(j)], version=v), txt))
        B = {k: {"n": tot[k], "caught_v1": caught[1][k], "caught_v2": caught[2][k]} for k in tot}
        run.log(f"B. guard sensitivity (corrupted template reports caught): {B}")
        missed = [(kind, txt) for j, (g, kind, txt) in enumerate(cor)
                  if not M.compare(M.expected(g["template"], g["graph"]["no_acute_abnormality"]), M.claims(parsed[str(j)], version=2), txt)]
        for kind, txt in missed[:10]:
            run.log(f"  v2 MISSED {kind}: {txt}")
        # C. does a retry repair a GENUINE error? The model is shown a corrupted report as its previous answer, plus
        # the feedback (v1 = names the finding only, as in run 20261007-093237; v2 = quotes the template sentence).
        from stage8_phrase import feedback as feedback_v1, split_sections
        C = {}
        pairs = []
        for j, (g, kind, txt) in enumerate(cor):
            if kind == "add_side" and not a.only_core:
                pass
            exp = M.expected(g["template"], g["graph"]["no_acute_abnormality"])
            mm = M.compare(exp, M.claims(parsed[str(j)], version=2), txt)
            if mm:
                pairs.append((g, kind, txt, exp, mm))
        pairs = pairs[: a.n_repair]
        from concurrent.futures import ThreadPoolExecutor
        from stage8_phrase import prompt_for
        for fbv in ("v1", "v2"):
            def one(x):
                g, kind, txt, exp, mm = x
                prev = f"FINDINGS: {txt}\nIMPRESSION: {g['template']['impression']}"
                fbt = feedback_v1(mm) if fbv == "v1" else S8.feedback(mm, g["template"])
                return llm(prompt_for(exp, g["graph"]["no_acute_abnormality"]), [{"role": "assistant", "content": prev}, {"role": "user", "content": fbt}], 0.3)
            with ThreadPoolExecutor(2) as ex:
                outs = list(ex.map(one, pairs))
            secs = [split_sections(o) if o else None for o in outs]
            texts = [f"{s_[0]} {s_[1]}" if s_ else "" for s_ in secs]
            pp = rg([t or "." for t in texts])
            fixed = Counter()
            tot_k = Counter()
            for i, (g, kind, txt, exp, mm) in enumerate(pairs):
                tot_k[kind] += 1
                fixed[kind] += bool(texts[i]) and not M.compare(exp, M.claims(pp[str(i)], version=2), texts[i])
            C[fbv] = {k: {"n": tot_k[k], "repaired": fixed[k]} for k in tot_k}
            run.log(f"C. retry repairs a genuine error, feedback {fbv}: {C[fbv]}")
        (run.dir / "summary.json").write_text(json.dumps({"A": A, "B": B, "C": C, "v2_missed": missed}, indent=1, default=str))


if __name__ == "__main__":
    main()
