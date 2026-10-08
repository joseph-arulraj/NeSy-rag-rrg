"""Readable end-to-end traces for a few validate studies of a runner output (for human review).
Per study: reference report (RadGraph-tokenised validate reference, cached), per-finding probability / band /
side / zone / feature-group contributions, the full audit trail (every rule that fired), case evidence, template,
the LLM prompt, every LLM attempt with its comparator result, the final report and its source.
Picks one study per category; validate only (seen by backbone)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]

CATS = [
    ("Normal call", lambda g, s: g["no_acute_abnormality"]),
    ("One stated finding", lambda g, s: n_stated(g) == 1),
    ("Several findings with side and zone", lambda g, s: n_stated(g) >= 3 and any(n.get("anatomy") for n in g["nodes"].values())),
    ("Side withheld by R2 (side head and regions disagree)", lambda g, s: any(a["rule"] == "R2" for a in g["audit"])),
    ("Parent raised by D2", lambda g, s: any(a["rule"] == "D2" for a in g["audit"])),
    ("Only 'possible' findings", lambda g, s: n_stated(g) >= 1 and all(n["band"] != "present" for f, n in g["nodes"].items() if f != "any_abnormality")),
    ("Not normal, nothing reportable ('No finding meets the reporting threshold')", lambda g, s: not g["no_acute_abnormality"] and n_stated(g) == 0),
    ("LLM first attempt failed, rescued by a retry", lambda g, s: s and s["n_attempts"] > 1 and not s["fallback"]),
]


def n_stated(g):
    return sum(n["band"] in ("present", "possible") for f, n in g["nodes"].items() if f != "any_abnormality")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/20261007-120046_pipeline-val-v2")
    ap.add_argument("--refs", default="data/features/radgraph_val_refs")
    ap.add_argument("--out", default="outputs/trace_samples_val.md")
    a = ap.parse_args()
    run = V2 / a.run
    G = {json.loads(l)["study_id"]: json.loads(l) for l in open(run / "graphs.jsonl")}
    S = {json.loads(l)["study_id"]: json.loads(l) for l in open(run / "stage8.jsonl")}
    refs = {}
    for fp in sorted((V2 / a.refs).glob("*.jsonl.gz")):
        for l in gzip.open(fp, "rt"):
            r = json.loads(l)
            refs[r["study_id"]] = r["text"]
    md = ["# End-to-end traces, validate (seen by backbone)", "",
          f"Runner output `v2/{a.run}` (C+A+R head; R1 off, R2 on; comparator v2). Reference reports are the validate "
          "radiologist reports as tokenised by RadGraph (punctuation spaced); they were never shown to the model or the LLM.", "",
          "How to read a trace: **probabilities** come from the head; **bands** from the thresholds (rule D1); every later "
          "change is listed in the **audit trail** with the rule that made it; the **template** is built from the graph by "
          "rules P1-P8; the **LLM** sees only the prompt shown; its text is accepted only if the RadGraph check finds "
          "exactly the template's statements, otherwise it retries (up to 3) and then falls back to the template.", ""]
    used = set()
    for title, cond in CATS:
        sid = next((s for s, g in G.items() if s not in used and cond(g["graph"], S.get(s))), None)
        if sid is None:
            md += [f"## {title}", "", "_no study in this category_", ""]
            continue
        used.add(sid)
        g, t, s = G[sid]["graph"], G[sid]["template"], S.get(sid)
        md += [f"## {title}: study {sid}", "", "**Reference report (not used by the pipeline):**", "", f"> {refs.get(sid, '(not available)')}", "",
               "**Belief graph** (probability from the head; band; side with side-head confidence; zone):", "",
               "| finding | p | band | side (conf) | zone |", "|---|---|---|---|---|"]
        for f, n in g["nodes"].items():
            z = ", ".join(f"{x.get('side')} {x.get('zone')}" for x in (n.get("anatomy") or []) if x.get("zone")) or ""
            sc = f"{n['side']} ({n['side_conf']:.2f})" if n.get("side") and n.get("side_conf") is not None else (n.get("side") or "")
            md.append(f"| {f} | {n['prob']:.3f} | **{n['band']}** | {sc} | {z} |" if n.get("prob") is not None else f"| {f} | - | {n['band']} | | |")
        md += ["", f"Normal call (D3): **{g['no_acute_abnormality']}**", "", "**Audit trail** (D1 band assignments summarised; every other rule in full):", ""]
        d1 = [x for x in g["audit"] if x["rule"] == "D1"]
        md.append(f"- D1 x{len(d1)}: " + "; ".join(f"{x['target']}={x['after']}" for x in d1))
        for x in g["audit"]:
            if x["rule"] != "D1":
                md.append(f"- **{x['rule']}** {x['target']}.{x['field']}: {x['before']!s} -> {x['after']!s} ({x['reason']})")
        if g.get("cases"):
            pos = {f: v["positive"] for f, v in g["cases"]["counts"].items() if v["positive"]}
            md += ["", f"**Case evidence** (context only, never changes a band; not in the report since show_case_evidence is off): "
                       f"{g['cases']['k']} similar fit-split studies; positives among them: {pos}"]
        md += ["", "**Template report** (rules P1-P8):", "", f"FINDINGS: {t['findings']}", "", f"IMPRESSION: {t['impression']}", ""]
        if t.get("omitted"):
            md.append("Omitted from the text: " + "; ".join(f"{o['finding']} ({o['rule']}: {o['reason']})" for o in t["omitted"]))
        if s:
            md += ["", "**Prompt sent to the LLM** (the only input it gets):", "", "```", s["prompt"], "```", ""]
            for k, at in enumerate(s["attempts"], 1):
                mm = "; ".join(f"{m['kind']} {m.get('finding')}" for m in at["mismatch"]) or "none (accepted)"
                md += [f"- Attempt {k}: {at.get('text')}", f"  - comparator mismatches: {mm}"]
            md += ["", f"**Final report** (source: {s['source']}):", "", f"FINDINGS: {s['final_text']['findings']}", "",
                   f"IMPRESSION: {s['final_text']['impression']}", ""]
        md.append("---")
    out = V2 / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(out, len(used), "studies")


if __name__ == "__main__":
    main()
