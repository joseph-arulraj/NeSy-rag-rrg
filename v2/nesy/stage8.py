"""Stage 8 phrasing with a RadGraph round-trip guard, as a library for the end-to-end runner.

Same prompt, system message and LLM client as scripts/stage8_phrase.py (run 20261007-093237_stage8-car); the
retry policy and comparator version are configuration (configs/pipeline.yaml: stage8). The prompt carries only
the template's statements: no study or patient ID, no report text, no case evidence. The key is read at run time
and never logged. The pipeline always returns a report: after the last failed attempt, the template.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import kg, rg_match as M

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage8_phrase import SYSTEM, LLM, prompt_for, split_sections  # noqa: E402,F401  (frozen prompt and client)

def feedback(mism: list[dict], template: dict | None = None) -> str:
    """Retry message (v2): names each mismatch and quotes the template's own sentence for that finding, so the
    model is told exactly what to write (v1 only named the finding, and the model repeated its text)."""
    by = kg.load_findings()["_by_id"]
    tsent = {s["finding"]: s["text"] for s in (template or {}).get("sentences", [])}
    msgs = []
    for m in mism:
        f = m.get("finding")
        lab = by[f]["label"].lower() if f else ""
        k, e = m["kind"], m.get("expected")
        use = f" Write it as: \"{tsent[f]}\"" if f in tsent else ""
        if k == "omitted":
            msgs.append(f"the statement about '{lab}' ({e}) is missing or not recognisable; include it as its own sentence.{use}")
        elif k == "added":
            msgs.append(f"remove every mention of '{lab}': it is not in the list")
        elif k == "added_negative":
            msgs.append(f"remove the negative statement about '{lab}': it is not in the list")
        elif k in ("polarity", "certainty"):
            msgs.append(f"'{lab}' must be stated as {e}.{use}")
        elif k == "side":
            msgs.append((f"the side of '{lab}' must be {e}." if e else
                         f"do not give any side for '{lab}', and do not mention it in a sentence that names a side.") + use)
        elif k == "zone":
            msgs.append(f"the zone of '{lab}' must be {e}." if e else f"do not give any zone for '{lab}'.")
        elif k == "lobe":
            msgs.append("do not mention any lung lobe")
        elif k == "malformed":
            msgs.append("use exactly the form 'FINDINGS: ...' then 'IMPRESSION: ...'")
    return ("Your previous report broke these rules: " + " ".join(msgs) + " Rewrite the whole report following the "
            "statement list exactly, one statement per sentence, and do not summarise several findings in one sentence.")


def phrase(items: list[dict], llm, rg, max_attempts: int, threads: int, retry: str, comparator: str,
           log=print) -> list[dict]:
    """items: [{study_id, template: render() output, no_acute}] -> one result dict per item (template check included).
    comparator: 'v1' | 'v2' (rg_match.claims version). retry: 'feedback' | 'none'."""
    ver = 1 if comparator == "v1" else 2
    for it in items:
        it["exp"] = M.expected(it["template"], it["no_acute"])
        it["prompt"] = prompt_for(it["exp"], it["no_acute"])
        it["attempts"], it["history"] = [], []
    ttxt = [f"{it['template']['findings']} {it['template']['impression']}" for it in items]
    tp = rg(ttxt)
    for k, it in enumerate(items):
        it["template_claims"] = M.claims(tp[str(k)], version=ver)
        it["template_entities"] = tp[str(k)].get("entities", {})
        it["template_mismatch"] = M.compare(it["exp"], it["template_claims"], ttxt[k])
    attempts = max_attempts if retry == "feedback" else 1
    pending = list(range(len(items)))
    for att in range(attempts):
        if not pending:
            break
        with ThreadPoolExecutor(threads) as ex:
            outs = list(ex.map(lambda k: llm(items[k]["prompt"], items[k]["history"], 0.0 if att == 0 else 0.3), pending))
        texts, keys = [], []
        for k, o in zip(pending, outs):
            sec = split_sections(o) if o else None
            if sec is None:
                items[k]["attempts"].append({"text": o, "raw": o, "mismatch": [{"kind": "malformed" if o else "llm_error", "finding": None}]})
                continue
            texts.append(f"{sec[0]} {sec[1]}")
            keys.append(k)
            items[k]["attempts"].append({"text": texts[-1], "raw": o, "sections": {"findings": sec[0], "impression": sec[1]}})
        parsed = rg(texts) if texts else {}
        for j, k in enumerate(keys):
            cl = M.claims(parsed[str(j)], version=ver)
            items[k]["attempts"][-1]["claims"] = cl
            items[k]["attempts"][-1]["entities"] = parsed[str(j)].get("entities", {})
            items[k]["attempts"][-1]["mismatch"] = M.compare(items[k]["exp"], cl, texts[j])
        still = []
        for k in pending:
            last = items[k]["attempts"][-1]
            if last["mismatch"]:
                if last.get("raw"):
                    items[k]["history"] = [{"role": "assistant", "content": last["raw"]},
                                           {"role": "user", "content": feedback(last["mismatch"], items[k]["template"])}]
                still.append(k)
        pending = still
    out = []
    for it in items:
        ok = [x for x in it["attempts"] if not x["mismatch"]]
        final = ok[0]["sections"] if ok else {"findings": it["template"]["findings"], "impression": it["template"]["impression"]}
        final_claims = ok[0]["claims"] if ok else it["template_claims"]
        final_entities = ok[0]["entities"] if ok else it["template_entities"]
        out.append({"study_id": it["study_id"], "n_attempts": len(it["attempts"]),
                    "first_pass_match": bool(it["attempts"] and not it["attempts"][0]["mismatch"]),
                    "fallback": not ok, "source": "template" if not ok else "llm", "final_text": final,
                    "template_text": {"findings": it["template"]["findings"], "impression": it["template"]["impression"]},
                    "final_claims": final_claims, "final_entities": final_entities,
                    "template_mismatch": it["template_mismatch"], "attempts": it["attempts"], "expected": it["exp"],
                    "prompt": it["prompt"]})
    return out
