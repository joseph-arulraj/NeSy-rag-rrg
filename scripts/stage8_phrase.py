"""Stage 8 (brief stage 7, second part): phrase each belief graph with the language model, parse the
generated report with RadGraph, compare it with the graph, regenerate on mismatch, fall back to the
template after --max-attempts failures. The pipeline always returns a report.

The prompt carries only what the graph licenses (the template's statements: finding label, certainty,
side, zone, and whether the study is called normal). No study or patient ID, no report text, no case-based
evidence. LLM: KCL-hosted OpenAI-compatible endpoint (configs/llm.yaml llm.base_url / llm.model); the
key is read from --key-file and never logged.

Also measured, as a check on the comparator itself: the round-trip match rate of the template reports
(the template states exactly the graph, so its mismatches are parser / lexicon limits).

Checkpoint: results.jsonl (one line per finished study); a restart skips finished studies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, kg, paths as P, pipeline_graph as PG, region_support as RS, report as R, rg_match as M  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

SYSTEM = (
    "You are a radiologist writing a chest X-ray report. You are given the complete list of statements the "
    "report may make. Write the report with two sections, exactly in this form:\n"
    "FINDINGS: <sentences>\nIMPRESSION: <sentences>\n"
    "Rules: write exactly one statement per sentence, in FINDINGS and in IMPRESSION (never list several findings "
    "in one sentence). Write every 'possible' statement as a sentence that begins with 'Possible'. Write support devices as "
    "'Support devices are in place.' (or 'Possible support device.'). Include every "
    "listed statement and nothing else. Do not add any finding, negative finding, "
    "measurement, comparison, recommendation, history or device detail that is not listed. Keep each "
    "statement's certainty: 'present' is stated plainly, 'possible' is hedged (for example 'possible' or "
    "'may represent'), 'absent' is stated as 'no ...'. Mention a side or zone only where one is given, and "
    "never mention a lung lobe. If the study is marked normal, the impression is 'No acute cardiopulmonary "
    "abnormality.' If it is NOT marked normal, never write that the study is normal or that there is no acute "
    "abnormality; if it is not marked normal and no statement is present or possible, the impression is 'No finding "
    "meets the reporting threshold.' Use plain clinical English.")


def prompt_for(exp: dict, normal: bool | None) -> str:
    by = kg.load_findings()["_by_id"]
    lines = []
    for f, v in exp.items():
        if f == "any_abnormality":
            continue
        s = f"- {by[f]['label'].lower()}: {v['cert']}"
        if v.get("side"):
            s += f"; side: {v['side']}"
        if v.get("zone"):
            s += f"; zone: {v['zone']}"
        lines.append(s)
    head = "Study marked normal: " + ("yes" if normal else "no")
    return head + "\nStatements:\n" + ("\n".join(lines) if lines else "- none")


def feedback(mism: list[dict]) -> str:
    by = kg.load_findings()["_by_id"]
    msgs = []
    for m in mism:
        lab = by[m["finding"]]["label"].lower() if m.get("finding") else ""
        k = m["kind"]
        msgs.append({"omitted": f"'{lab}' ({m.get('expected')}) is missing",
                     "added": f"'{lab}' was added but is not in the list",
                     "added_negative": f"a negative statement about '{lab}' was added but is not in the list",
                     "polarity": f"'{lab}' must be {m.get('expected')}",
                     "certainty": f"'{lab}' must be {m.get('expected')}",
                     "side": f"the side of '{lab}' must be {m.get('expected') or 'not stated'}",
                     "zone": f"the zone of '{lab}' must be {m.get('expected') or 'not stated'}",
                     "lobe": "do not mention any lobe"}[k])
    return "Your previous report broke these rules: " + "; ".join(msgs) + ". Rewrite it following the list exactly."


class LLM:
    def __init__(self, key_file: str, cfg: dict):
        self.key = Path(key_file).read_text().strip()
        self.url, self.model = cfg["base_url"], cfg["model"]
        self.calls = self.errors = 0
        self.status: dict = {}                                  # HTTP status / exception counts (never the key)

    def __call__(self, user: str, history: list | None = None, temperature: float = 0.0) -> str | None:
        import requests
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}] + (history or [])
        for attempt in range(6):
            try:
                self.calls += 1
                r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, timeout=90,
                                  json={"model": self.model, "messages": msgs, "temperature": temperature, "max_tokens": 500})
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                self.status[r.status_code] = self.status.get(r.status_code, 0) + 1
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt * 3)
                    continue
                self.errors += 1
                return None
            except Exception as exc:                            # noqa: BLE001
                self.status[type(exc).__name__] = self.status.get(type(exc).__name__, 0) + 1
                time.sleep(2 ** attempt * 3)
        self.errors += 1
        return None


def split_sections(txt: str) -> tuple[str, str] | None:
    m = re.search(r"FINDINGS:\s*(.*?)\s*IMPRESSION:\s*(.*)", txt or "", re.S | re.I)
    if not m:
        return None
    return " ".join(m.group(1).split()), " ".join(m.group(2).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-run", required=True, help="final head run")
    ap.add_argument("--rules-run", required=True, help="rules_eval run (rules_eval.json, region_thresholds.json)")
    ap.add_argument("--s4-run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=3, help="LLM attempts before falling back to the template")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2, help="concurrent LLM calls (4 hit HTTP 429 rate limits)")
    ap.add_argument("--key-file", default=str(Path.home() / ".rrg_llm_key"))
    ap.add_argument("--out", default=None, help="results directory (default: run dir); a fixed one resumes across runs")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load((P.V2 / "configs/llm.yaml").read_text())["llm"]
    with Run("stage8-phrase", {**vars(args), "llm_url": cfg["base_url"], "llm_model": cfg["model"]}, args.run_dir) as run:
        out_dir = Path(args.out) if args.out else run.dir
        out_dir.mkdir(parents=True, exist_ok=True)
        res_path = out_dir / "results.jsonl"
        done = set()
        if res_path.exists():
            done = {json.loads(l)["study_id"] for l in res_path.read_text().splitlines() if l.strip()}
        rules = json.loads((Path(args.rules_run) / "rules_eval.json").read_text())
        use_r1, use_r2 = rules["R1"]["keep"], rules["R2"]["keep"]
        rthr = rules["region_thresholds"]
        pr = Path(args.pred_run)
        thr = json.loads((pr / "thresholds_4band.json").read_text())
        preds = pd.read_parquet(pr / "predictions_non_test.parquet")
        t = D.study_table()[["study_id", "dicom_id"]]
        df = preds[preds.split == args.split].merge(t, on="study_id")
        if args.limit:
            df = df.sample(args.limit, random_state=0) if args.limit < len(df) else df
        todo = df[~df.study_id.isin(done)]
        FIND, idx, sc = RS.load(args.s4_run)
        pos = {d: i for i, d in enumerate(idx.dicom_id)}
        run.log(f"{args.split}: {len(df):,} studies, already done {len(done):,}, to do {len(todo):,}; rules R1={use_r1} R2={use_r2}; "
                f"LLM {cfg['model']} at {cfg['base_url']}; max attempts {args.max_attempts}")
        llm = LLM(args.key_file, cfg)
        from radgraph import RadGraph
        rg = RadGraph(model_type="modern-radgraph-xl", batch_size=32, cuda=0)
        prog = Progress(run, len(todo), "studies", every_s=120)
        n = 0
        for a in range(0, len(todo), args.chunk):
            part = todo.iloc[a:a + args.chunk]
            items = []
            for row in part.to_dict("records"):
                i = pos.get(row["dicom_id"])
                ev = {} if i is None else {f: RS.evidence(np.asarray(sc[i], np.float32), fi, f) for fi, f in enumerate(FIND) if f in D.FINDINGS}
                g = PG.build(row["study_id"], row, thr, ev, rthr, use_r1=use_r1, use_r2=use_r2)
                tpl = R.render(g)
                exp = M.expected(tpl, g.no_acute_abnormality)
                items.append({"study_id": int(row["study_id"]), "graph": g, "template": tpl, "exp": exp,
                              "prompt": prompt_for(exp, g.no_acute_abnormality), "attempts": [], "history": []})
            # template round trip (comparator check)
            # RadGraph is parsed WITHOUT the section headers (they make it drop entities; checked 2026-10-07)
            ttxt = [f"{it['template']['findings']} {it['template']['impression']}" for it in items]
            tp = rg(ttxt)
            for k, it in enumerate(items):
                it["template_mismatch"] = M.compare(it["exp"], M.claims(tp[str(k)]), ttxt[k])
            pending = list(range(len(items)))
            for att in range(args.max_attempts):
                if not pending:
                    break
                with ThreadPoolExecutor(args.threads) as ex:
                    outs = list(ex.map(lambda k: llm(items[k]["prompt"], items[k]["history"], 0.0 if att == 0 else 0.3), pending))
                texts, keys = [], []
                for k, o in zip(pending, outs):
                    sec = split_sections(o) if o else None
                    if sec is None:
                        items[k]["attempts"].append({"text": o, "mismatch": [{"kind": "malformed" if o else "llm_error", "finding": None}]})
                        continue
                    texts.append(f"{sec[0]} {sec[1]}")
                    keys.append(k)
                parsed = rg(texts) if texts else {}
                for j, k in enumerate(keys):
                    mm = M.compare(items[k]["exp"], M.claims(parsed[str(j)]), texts[j])
                    items[k]["attempts"].append({"text": texts[j], "raw": outs[pending.index(k)],
                                                 "sections": dict(zip(("findings", "impression"), split_sections(outs[pending.index(k)]))),
                                                 "mismatch": mm})
                still = []
                for k in pending:
                    last = items[k]["attempts"][-1] if items[k]["attempts"] else None
                    if last is None or last["mismatch"]:
                        if last is not None and last.get("text"):
                            items[k]["history"] = [{"role": "assistant", "content": last.get("raw") or last["text"]},
                                                   {"role": "user", "content": feedback([m for m in last["mismatch"] if m["kind"] not in ("malformed", "llm_error")]) if any(m["kind"] not in ("malformed", "llm_error") for m in last["mismatch"]) else "Use exactly the FINDINGS: / IMPRESSION: format."}]
                        still.append(k)
                pending = still
            with open(res_path, "a") as fh:
                for it in items:
                    ok = [x for x in it["attempts"] if not x["mismatch"]]
                    final = ok[0]["sections"] if ok else {"findings": it["template"]["findings"], "impression": it["template"]["impression"]}
                    fh.write(json.dumps({"study_id": it["study_id"], "n_attempts": len(it["attempts"]),
                                         "first_pass_match": bool(it["attempts"] and not it["attempts"][0]["mismatch"]),
                                         "fallback": not ok, "final_text": final, "template_mismatch": it["template_mismatch"],
                                         "attempts": it["attempts"], "expected": it["exp"], "prompt": it["prompt"],
                                         "graph": it["graph"].to_dict()}, default=str) + "\n")
            n += len(items)
            prog.update(n)
        # ---- summary over everything in results.jsonl for this split's studies
        rs = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()]
        rs = [r for r in rs if r["study_id"] in set(df.study_id)]
        N = len(rs)
        first = sum(r["first_pass_match"] for r in rs)
        regen = sum(r["n_attempts"] > 1 for r in rs)
        fb = sum(r["fallback"] for r in rs)
        tmatch = sum(not r["template_mismatch"] for r in rs)
        kinds_first = Counter(m["kind"] for r in rs if r["attempts"] for m in r["attempts"][0]["mismatch"])
        kinds_all = Counter(m["kind"] for r in rs for a in r["attempts"] for m in a["mismatch"])
        kinds_tpl = Counter(m["kind"] for r in rs for m in r["template_mismatch"])
        fk = Counter((m["kind"], m.get("finding")) for r in rs if r["attempts"] for m in r["attempts"][0]["mismatch"])
        summ = {"n": N, "first_pass_match_rate": first / N, "regeneration_rate": regen / N, "fallback_rate": fb / N,
                "final_llm_match_rate": 1 - fb / N, "template_roundtrip_match_rate": tmatch / N,
                "mismatch_kinds_first_attempt": dict(kinds_first.most_common()), "mismatch_kinds_all_attempts": dict(kinds_all.most_common()),
                "template_mismatch_kinds": dict(kinds_tpl.most_common()),
                "top_first_attempt_mismatches": [f"{k}:{f}={c}" for (k, f), c in fk.most_common(15)],
                "llm_calls": llm.calls, "llm_errors": llm.errors, "llm_status_counts": {str(k): v for k, v in llm.status.items()}}
        (run.dir / "summary.json").write_text(json.dumps(summ, indent=2))
        # three sample reports with their belief graphs: normal call, one stated finding, several stated findings
        def n_stated(r):
            return sum(v["cert"] in ("present", "possible") for f, v in r["expected"].items())
        picks, md = [], ["# Stage 8 sample reports (validate; seen by backbone)", ""]
        for name, cond in (("normal call", lambda r: r["graph"].get("no_acute_abnormality")),
                           ("one stated finding", lambda r: n_stated(r) == 1),
                           ("several stated findings", lambda r: n_stated(r) >= 3)):
            c = [r for r in rs if cond(r) and not r["fallback"]] or [r for r in rs if cond(r)]
            if not c:
                continue
            r = c[0]
            picks.append(r["study_id"])
            nodes = r["graph"]["nodes"]
            md += [f"## {name}: study {r['study_id']} ({'LLM, ' + str(r['n_attempts']) + ' attempt(s)' if not r['fallback'] else 'template fallback'})", "",
                   "Belief graph (finding | p | band | side | zone):", "```"]
            for f, n in nodes.items():
                if n["band"] != "silent":
                    z = (n.get("anatomy") or [{}])[0].get("zone") if n.get("anatomy") else None
                    md.append(f"{f:28s} {n['prob'] if n['prob'] is None else round(n['prob'], 3)!s:7s} {n['band']:9s} {n.get('side')!s:10s} {z}")
            md += [f"no_acute_abnormality: {r['graph'].get('no_acute_abnormality')}", "```", "",
                   f"FINDINGS: {r['final_text']['findings']}", "", f"IMPRESSION: {r['final_text']['impression']}", ""]
        (run.dir / "samples.md").write_text("\n".join(md))
        run.log(f"samples.md written (studies {picks})")
        run.log("SUMMARY " + json.dumps(summ, indent=1))
        run.metric(summary="stage8", **{k: v for k, v in summ.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
