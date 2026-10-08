"""Stage 8 round-trip guard: map a RadGraph parse of a report to (finding, certainty, side, zone) claims
and compare them with the statements the belief graph licenses (the template's sentences).

Certainty: RadGraph "definitely present" -> present, "uncertain" -> possible, "definitely absent" ->
absent; "normal" / "within normal limits" / "unremarkable" observations of the heart or mediastinum are
absent claims; "borderline" is possible. Side and zone come from the observation's own tokens and the
anatomy entities it is linked to (located_at / modify, either direction).

Mismatch kinds:
  omitted         a licensed statement is not in the text
  added           the text claims a finding (present / possible) the graph does not state
  added_negative  the text negates a finding the graph does not state as a pertinent negative
  polarity        stated present/possible vs absent the other way round
  certainty       present vs possible
  side            the stated side differs from the graph's, or a side is given where the graph has none
  zone            the stated zone differs, or a zone is given where the graph has none
  lobe            any lobe is mentioned (lobes are never licensed)
A finding the text claims that is an is_a ANCESTOR of a stated finding is tolerated (e.g. "consolidative
opacity" for consolidation).
"""
from __future__ import annotations

import re

from . import kg

DEVICE = {"tube", "tubes", "line", "lines", "catheter", "catheters", "pacemaker", "pacer", "device", "devices", "wire", "wires",
          "port", "stent", "clip", "clips", "drain", "drains", "icd", "picc", "ett", "electrode", "electrodes", "leads", "lead",
          "hardware", "support"}
SIDE_WORDS = {"left": "left", "right": "right", "bilateral": "bilateral", "both": "bilateral", "bibasilar": "bilateral",
              "bibasal": "bilateral"}
ZONE_WORDS = {"upper": "upper", "apical": "upper", "apex": "upper", "mid": "middle", "middle": "middle", "lower": "lower",
              "basal": "lower", "base": "lower", "bases": "lower", "basilar": "lower"}
NORMALISH = {"normal", "unremarkable", "limits", "stable", "unchanged"}
HEART = {"heart", "cardiac", "heart size", "size"}
MEDIASTINUM = {"cardiomediastinal", "mediastinal", "mediastinum", "silhouette", "contours", "contour"}
MEDIASTINUM_V2 = MEDIASTINUM | {"cardiomediastinum"}


def _cert(label: str) -> str:
    s = label.split("::")[-1]
    return {"definitely present": "present", "uncertain": "possible", "definitely absent": "absent"}.get(s, "present")


HEDGE = {"possible", "possibly", "may", "might", "cannot", "likely", "probable", "probably", "suspected", "questionable",
         "borderline", "could", "suggest", "suggests", "suggestive"}
NEGATION = {"no", "without", "not", "negative", "absent"}


def _sentence_words(parsed: dict) -> list[set[str]]:
    """For each token index of RadGraph's tokenised text, the lower-cased words of its sentence."""
    toks = (parsed.get("text") or "").split()
    out, cur, start = [None] * len(toks), [], 0
    for i, w in enumerate(toks + ["."]):
        if i < len(toks):
            cur.append(w.lower())
        if i == len(toks) or w in (".", "!", "?") or w.endswith("."):
            ws = set(re.findall(r"[a-z]+", " ".join(cur)))
            for j in range(start, min(i + 1, len(toks))):
                out[j] = ws
            cur, start = [], i + 1
    return out


def _sentence_ids(parsed: dict) -> tuple[list[int], list[str]]:
    """Sentence index of each token, and each sentence's text."""
    toks = (parsed.get("text") or "").split()
    ids, texts, cur, sid = [], [], [], 0
    for w in toks:
        ids.append(sid)
        cur.append(w)
        if w in (".", "!", "?") or w.endswith("."):
            texts.append(" ".join(cur))
            cur, sid = [], sid + 1
    if cur:
        texts.append(" ".join(cur))
    return ids, texts


# v2 lexical reading of one sentence (used only for sentences in which RadGraph found no finding): same finding
# words and priority as the entity-level mapping below
_LEX = [("pneumothorax", r"pneumothora(x|ces)"), ("pleural_effusion", r"effusions?"),
        ("pleural_other", r"pleural (thickening|plaques?|scarring)"), ("edema", r"o?edema|congestion"),
        ("cardiomegaly", r"cardiomegaly"), ("atelectasis", r"atelecta(sis|tic)|collapse"),
        ("consolidation", r"consolidat(ion|ions|ive)"), ("pneumonia", r"pneumonia|infection|infectious"),
        ("lung_lesion", r"nodules?|mass(es)?|lesions?"), ("fracture", r"fractures?"),
        ("lung_opacity", r"opacit(y|ies)|opacifications?"),
        ("support_devices", r"(support )?devices?|tubes?|lines?|catheters?|pacemaker|pacer|wires?|drains?|picc|stents?|clips?|hardware"),
        ("enlarged_cardiomediastinum", r"cardiomediastin(al|um)( silhouette| contours?)?|mediastin(al|um)"),
        ("cardiomegaly", r"heart|cardiac"),
        ("any_abnormality", r"acute (cardiopulmonary )?(abnormality|process)")]


def _lexical(sentence: str) -> list[dict]:
    w = set(re.findall(r"[a-z]+", sentence.lower()))
    low = sentence.lower()
    for f, pat in _LEX:
        if not re.search(r"\b(" + pat + r")\b", low):
            continue
        if f in ("enlarged_cardiomediastinum", "cardiomegaly") and not re.search(r"cardiomegaly", low):
            if not w & {"enlarged", "enlargement", "large", "borderline", "normal", "limits", "unremarkable", "widened", "widening", "prominent", "size"}:
                continue
        cert = "absent" if w & NEGATION else "possible" if w & (HEDGE | {"borderline"}) else "present"
        if f in ("enlarged_cardiomediastinum", "cardiomegaly") and cert == "present" and w & NORMALISH:
            cert = "absent"
        if f == "any_abnormality" and cert == "present":
            continue
        side = next((SIDE_WORDS[x] for x in w if x in SIDE_WORDS), None)
        if {"left", "right"} <= {SIDE_WORDS[x] for x in w if x in SIDE_WORDS}:
            side = "bilateral"
        zone = next((ZONE_WORDS[x] for x in w if x in ZONE_WORDS), None)
        return [{"finding": f, "cert": cert, "side": side, "zone": zone, "tokens": sentence, "lexical": True}]
    return []


def claims(parsed: dict, version: int = 1) -> list[dict]:
    """parsed: RadGraph output for one report {text, entities}. -> list of {finding, cert, side, zone, tokens}.
    Side, zone and hedging are read from the observation's own sentence as well as its linked entities
    (generation is constrained to one statement per sentence, so the sentence is a reliable scope).
    version 2 (2026-10-07): (a) device and heart / mediastinum context words are also taken from the sentence
    (RadGraph often leaves "in place" / "normal" unlinked); (b) "cardiomediastinum" is a mediastinum word;
    (c) in a sentence that names several findings, side / zone / hedge words of the sentence are not attached to
    every finding (entity tokens and links only); (d) a sentence in which RadGraph found no finding at all is
    read lexically with the same finding words (RadGraph drops whole short sentences such as "Possible support
    device." or "No focal consolidation." depending on context)."""
    out = _claims_entities(parsed, version)
    if version >= 2:
        sid, texts = _sentence_ids(parsed)
        covered = {c["_sent"] for c in out if c.get("_sent") is not None}
        for k, t in enumerate(texts):
            if k not in covered:
                out += _lexical(t)
    for c in out:
        c.pop("_sent", None)
    return out


def _claims_entities(parsed: dict, version: int) -> list[dict]:
    ents = parsed.get("entities", {}) or {}
    sent = _sentence_words(parsed)
    sid, _ = _sentence_ids(parsed)
    links: dict[str, set[str]] = {k: set() for k in ents}
    for k, e in ents.items():
        for _, tgt in e.get("relations", []):
            if tgt in ents:
                links[k].add(tgt)
                links[tgt].add(k)
    out = []
    for k, e in ents.items():
        is_obs = e["label"].startswith("Observation")
        tok = e["tokens"].lower()
        if not is_obs:
            # devices are sometimes tagged as anatomy ("Support devices are present.")
            if set(re.findall(r"[a-z]+", tok)) & DEVICE:
                sw = sent[e["start_ix"]] if e.get("start_ix") is not None and e["start_ix"] < len(sent) and sent[e["start_ix"]] else set()
                c = "absent" if sw & NEGATION else "possible" if sw & HEDGE else "present"
                out.append({"finding": "support_devices", "cert": c, "side": None, "zone": None, "tokens": e["tokens"],
                            "_sent": _sid(sid, e)})
            continue
        words = set(re.findall(r"[a-z]+", tok))
        ctx = set()
        for j in links[k]:
            ctx |= set(re.findall(r"[a-z]+", ents[j]["tokens"].lower()))
            for j2 in links[j]:                             # one more hop: "left" -> "lung" -> observation
                if ents[j2]["label"].startswith("Anatomy"):
                    ctx |= set(re.findall(r"[a-z]+", ents[j2]["tokens"].lower()))
        allw = words | ctx
        sw0 = sent[e["start_ix"]] if e.get("start_ix") is not None and e["start_ix"] < len(sent) and sent[e["start_ix"]] else set()
        multi = False
        if version >= 2:
            allw = allw | (sw0 & (DEVICE | MEDIASTINUM | {"heart", "cardiac", "size", "cardiomediastinum"}))
            multi = _n_findings(sw0) > 1
        cert = _cert(e["label"])
        f = None
        if words & {"pneumothorax", "pneumothoraces"}:
            f = "pneumothorax"
        elif words & {"effusion", "effusions"}:
            f = "pleural_effusion"
        elif words & {"thickening", "plaque", "plaques", "scarring"} and ("pleural" in allw or "pleura" in allw):
            f = "pleural_other"
        elif words & {"edema", "oedema", "congestion"}:
            f = "edema"
        elif words & {"cardiomegaly"}:
            f = "cardiomegaly"
        elif words & {"atelectasis", "atelectatic", "collapse"}:
            f = "atelectasis"
        elif words & {"consolidation", "consolidations", "consolidative"}:
            f = "consolidation"
        elif words & {"pneumonia", "infection", "infectious"}:
            f = "pneumonia"
        elif words & {"nodule", "nodules", "mass", "masses", "lesion", "lesions"}:
            f = "lung_lesion"
        elif words & {"fracture", "fractures"}:
            f = "fracture"
        elif words & {"opacity", "opacities", "opacification", "opacifications"}:
            f = "lung_opacity"
        elif words & DEVICE:
            f = "support_devices"
        elif allw & DEVICE and words & {"present", "place", "seen", "noted", "position", "positioned", "appropriate", "unchanged", "stable"}:
            f = "support_devices"
        elif words & {"abnormality", "abnormalities", "process"} and (allw & {"acute", "cardiopulmonary"}):
            f = "any_abnormality"
        elif words & {"enlarged", "enlargement", "large", "borderline", "normal", "limits", "unremarkable", "widened", "widening", "prominent"}:
            if allw & (MEDIASTINUM_V2 if version >= 2 else MEDIASTINUM):
                f = "enlarged_cardiomediastinum"
            elif allw & {"heart", "cardiac", "size"}:
                f = "cardiomegaly"
            if f and words & NORMALISH and cert == "present":
                cert = "absent"
            if f and "borderline" in words and cert == "present":
                cert = "possible"
        if f is None:
            continue
        sw = sw0 if not multi else set()                         # v2: a list sentence's words do not scope every finding
        if cert == "present" and sw & HEDGE:
            cert = "possible"                                    # RadGraph often misses hedges ("There is possible X")
        if version >= 2 and cert in ("present", "possible") and not multi and sw & {"no", "without"}:
            cert = "absent"                                      # v2: RadGraph sometimes labels "No X is present." present
        scope = words | ctx | sw
        side = next((SIDE_WORDS[w] for w in scope if w in SIDE_WORDS), None)
        sides = {SIDE_WORDS[w] for w in scope if w in SIDE_WORDS}
        if {"left", "right"} <= sides:
            side = "bilateral"
        zone = next((ZONE_WORDS[w] for w in scope if w in ZONE_WORDS), None)
        out.append({"finding": f, "cert": cert, "side": side, "zone": zone, "tokens": e["tokens"], "_sent": _sid(sid, e)})
    return out


def _sid(sid: list[int], e: dict) -> int | None:
    i = e.get("start_ix")
    return sid[i] if i is not None and i < len(sid) else None


_FWORDS = [r"pneumothora", r"effusion", r"thickening|plaque", r"o?edema|congestion", r"cardiomegaly", r"atelecta|collapse",
           r"consolidat", r"pneumonia", r"nodule|mass|lesion", r"fracture", r"opacit|opacification",
           r"device|tube|line|catheter|pacemaker|wire|drain", r"cardiomediastin|mediastin", r"heart|cardiac"]


def _n_findings(words: set[str]) -> int:
    t = " ".join(sorted(words))
    return sum(bool(re.search(p, t)) for p in _FWORDS)


def expected(rendered: dict, no_acute: bool | None) -> dict[str, dict]:
    """Licensed statements from the template's sentences (P1-P8 already applied)."""
    exp = {}
    for s in rendered.get("sentences", []):
        exp[s["finding"]] = {"cert": s["band"], "side": s.get("side"), "zone": s.get("zone")}
    if no_acute:
        exp["any_abnormality"] = {"cert": "absent", "side": None, "zone": None}
    return exp


def compare(exp: dict[str, dict], found: list[dict], text: str) -> list[dict]:
    mism = []
    stated = {f for f, v in exp.items() if v["cert"] in ("present", "possible")}
    tolerated = set()
    for f in stated:
        tolerated |= set(kg.ancestors(f))
    by = {}
    for c in found:
        by.setdefault(c["finding"], []).append(c)
    for f, v in exp.items():
        cs = by.get(f, [])
        if not cs:
            mism.append({"kind": "omitted", "finding": f, "expected": v["cert"]})
            continue
        certs = {c["cert"] for c in cs}
        if v["cert"] not in certs:
            pos_e = v["cert"] in ("present", "possible")
            pos_f = bool(certs & {"present", "possible"})
            kind = "polarity" if pos_e != pos_f else "certainty"
            mism.append({"kind": kind, "finding": f, "expected": v["cert"], "found": sorted(certs)})
        if v["cert"] in ("present", "possible"):
            fs = {c["side"] for c in cs if c["side"]}
            if (v["side"] and v["side"] not in fs) or (not v["side"] and fs):
                mism.append({"kind": "side", "finding": f, "expected": v["side"], "found": sorted(fs)})
            fz = {c["zone"] for c in cs if c["zone"]}
            if (v["zone"] and v["zone"] not in fz) or (not v["zone"] and fz and f in ("lung_opacity", "atelectasis", "consolidation", "pneumonia", "lung_lesion")):
                mism.append({"kind": "zone", "finding": f, "expected": v["zone"], "found": sorted(fz)})
    for f, cs in by.items():
        if f in exp or f in tolerated:
            continue
        certs = {c["cert"] for c in cs}
        if certs & {"present", "possible"}:
            mism.append({"kind": "added", "finding": f, "found": sorted(certs)})
        else:
            mism.append({"kind": "added_negative", "finding": f})
    if re.search(r"\blobes?\b|\blingula", text, re.I):
        mism.append({"kind": "lobe", "finding": None})
    return mism
