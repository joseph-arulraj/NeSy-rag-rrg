#!/usr/bin/env python
"""Look at the 368,294 CLEAR concepts (needs only the CSV; no torch, no GPU).

  python scripts/explore_concepts.py                       # full profile + writes outputs/concept_exploration.{md,json}
  python scripts/explore_concepts.py --grep "left lower lobe" --max 25
  python scripts/explore_concepts.py --grep "^no " --sample 30
  python scripts/explore_concepts.py --config configs/hpc.yaml --set explore.seed=1

The category counts are regex heuristics over the raw phrases, so treat them as approximate.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.concepts.vocabulary import clean_text, load_concept_texts  # noqa: E402
from rrg.core.config import load_settings, require_file  # noqa: E402

# name -> (description, regex). Case-insensitive, matched on the whitespace-cleaned phrase.
CATEGORIES: dict[str, tuple[str, str]] = {
    "laterality": ("left/right/bilateral wording", r"\b(left|right|bilateral|bilaterally|bibasilar|bibasal|both)\b"),
    "lung_anatomy": ("lung / lobe / hilum / base / apex", r"\b(lung|lungs|lobe|lobes|pulmonary|perihilar|hilar|hila|hilum|apex|apical|basilar|base|bases|lingula|parenchym\w*)\b"),
    "pleura_diaphragm_wall": ("pleura / diaphragm / costophrenic / chest wall", r"\b(pleura\w*|costophrenic|diaphragm\w*|hemidiaphragm\w*|chest wall|thorax|thoracic)\b"),
    "heart_mediastinum": ("heart / mediastinum / aorta / airway", r"\b(cardiac|heart|cardiomegaly|cardiomediastinal|mediastin\w*|aort\w*|trachea\w*|carina|vascular|vasculature)\b"),
    "bone": ("ribs / spine / clavicle / bones", r"\b(rib|ribs|spine|spinal|vertebra\w*|clavicle\w*|scapula\w*|sternum|sternal|humer\w*|osseous|bone|bony)\b"),
    "finding_terms": ("common finding words", r"\b(effusion\w*|opacit\w*|consolidat\w*|atelectasis|atelectatic|edema|pneumonia|pneumothora\w*|nodul\w*|mass\w*|infiltrat\w*|fracture\w*|emphysema|scarring|fibrosis|cardiomegaly|enlarge\w*|congestion|granuloma\w*|calcif\w*)\b"),
    "devices": ("tubes / lines / hardware", r"\b(tube|catheter|line|lines|pacemaker|pacer|defibrillator|icd|wire|wires|port|drain|stent|clip|clips|picc|cvc|ngt|ett|nasogastric|endotracheal|tracheostomy|sternotomy|prosthe\w*|device|leads?|electrode\w*|balloon|valve)\b"),
    "negation": ("no / without / absent", r"\b(no|not|without|negative|absent|absence|free of)\b"),
    "normal_unremarkable": ("normal / clear / unremarkable", r"\b(normal|unremarkable|clear|within normal limits|intact)\b"),
    "temporal_comparison": ("refers to a prior study / change over time", r"\b(unchanged|interval|previous\w*|prior|compared|comparison|since|new|newly|increas\w*|decreas\w*|resolv\w*|improv\w*|worsen\w*|stable|persist\w*|again|progress\w*|earlier|recent|redemonstrated|re-demonstrated|remain\w*)\b"),
    "uncertainty": ("hedged / possible", r"\b(possible|possibly|probable|probably|may|might|could|likely|suggest\w*|concern\w*|cannot|questionable|versus|vs|perhaps|suspect\w*|equivocal|consider\w*)\b"),
    "severity_size": ("severity / size / distribution", r"\b(mild|mildly|moderate|moderately|severe|severely|small|tiny|large|massive|minimal|trace|marked|markedly|slight\w*|subtle|extensive|focal|patchy|diffuse)\b"),
    "numbers_measurements": ("digits or cm/mm", r"(\d|\b(cm|mm)\b)"),
    "sentence_like": ("contains a verb-ish word (is/are/seen/noted/...)", r"\b(is|are|was|were|has|have|been|being|appears?|seen|noted|identified|visuali[sz]ed|demonstrated|present)\b"),
}
_COMPILED = {k: re.compile(rx, re.IGNORECASE) for k, (_, rx) in CATEGORIES.items()}
STOP = set("the of and in is are a an to with at or for on there be by as from that this it its has have was were".split())
WORD_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 15), (16, 20), (21, 10**9)]


def pct(x: int, n: int) -> str:
    return f"{100.0 * x / n:5.1f}%"


def quantile(sorted_vals: list[int], q: float) -> float:
    return float(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))])


def build_report(texts: tuple[str, ...], seed: int, sample_size: int, top_tokens: int, per_cat: int) -> tuple[str, dict]:
    n = len(texts)
    rng = random.Random(seed)
    cleaned = [clean_text(t) for t in texts]
    words = [len(c.split()) for c in cleaned]
    sw = sorted(words)
    chars = sorted(len(c) for c in cleaned)
    dup_exact = n - len(set(texts))
    dup_norm = n - len({c.casefold() for c in cleaned})
    empty = sum(1 for c in cleaned if not c)
    multiline = [i for i, t in enumerate(texts) if "\n" in t]
    non_ascii = sum(1 for c in cleaned if not c.isascii())

    stats: dict = {
        "n_concepts": n, "duplicates_exact": dup_exact, "duplicates_case_whitespace_normalised": dup_norm,
        "empty": empty, "with_embedded_newline": len(multiline), "non_ascii": non_ascii,
        "words": {"min": sw[0], "mean": round(statistics.fmean(words), 2), "median": statistics.median(sw),
                  "p05": quantile(sw, 0.05), "p95": quantile(sw, 0.95), "max": sw[-1]},
        "chars": {"median": statistics.median(chars), "p95": quantile(chars, 0.95), "max": chars[-1]},
        "categories": {},
    }
    L: list[str] = []
    L += [f"# CLEAR concept vocabulary: {n:,} concepts", "",
          "## Basic facts", "",
          f"- concepts: **{n:,}**  |  exact duplicates: {dup_exact:,}  |  duplicates after case/whitespace normalisation: {dup_norm:,}",
          f"- empty: {empty}  |  with an embedded line break: {len(multiline)}  |  non-ASCII: {non_ascii}",
          f"- words per concept: min {sw[0]}, median {statistics.median(sw)}, mean {statistics.fmean(words):.1f}, p95 {quantile(sw, .95):.0f}, max {sw[-1]}",
          f"- characters per concept: median {statistics.median(chars)}, p95 {quantile(chars, .95):.0f}, max {chars[-1]}", "",
          "## Length (words)", "", "| words | concepts | share |", "|---|---|---|"]
    for lo, hi in WORD_BUCKETS:
        c = sum(1 for w in words if lo <= w <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")
        L.append(f"| {label} | {c:,} | {pct(c, n).strip()} |")

    # ---- categories
    hits: dict[str, list[int]] = {k: [] for k in CATEGORIES}
    for i, c in enumerate(cleaned):
        for k, rx in _COMPILED.items():
            if rx.search(c):
                hits[k].append(i)
    L += ["", "## What the phrases talk about (regex heuristics, a phrase can fall in several rows)", "",
          "| category | meaning | concepts | share |", "|---|---|---|---|"]
    for k, (desc, _) in CATEGORIES.items():
        L.append(f"| `{k}` | {desc} | {len(hits[k]):,} | {pct(len(hits[k]), n).strip()} |")
        stats["categories"][k] = len(hits[k])
    lat = set(hits["laterality"])
    anat = set(hits["lung_anatomy"]) | set(hits["pleura_diaphragm_wall"]) | set(hits["heart_mediastinum"]) | set(hits["bone"])
    anywhere = anat | set(hits["finding_terms"]) | set(hits["devices"])
    stats["laterality_and_anatomy"] = len(lat & anat)
    stats["no_anatomy_finding_or_device_term"] = n - len(anywhere)
    stats["temporal_comparison_share"] = round(len(hits["temporal_comparison"]) / n, 4)
    L += ["", f"- laterality word **and** an anatomy word: {len(lat & anat):,} ({pct(len(lat & anat), n).strip()})  <- what the RadLex anatomy/laterality tagging can act on",
          f"- laterality word but no anatomy word: {len(lat - anat):,}",
          f"- none of anatomy / finding / device vocabulary matched: {n - len(anywhere):,} ({pct(n - len(anywhere), n).strip()})",
          f"- **temporal/comparison wording: {len(hits['temporal_comparison']):,} ({pct(len(hits['temporal_comparison']), n).strip()})** "
          "-- these describe change against a prior exam, which this pipeline has no evidence to support (docs/pipeline.md GAP-11)"]

    # ---- tokens
    tok = Counter(w for c in cleaned for w in re.findall(r"[a-z][a-z\-']+", c.lower()) if w not in STOP)
    first = Counter((c.split()[0].lower() if c else "") for c in cleaned)
    L += ["", f"## Most frequent words (top {top_tokens}, stop-words removed)", "",
          ", ".join(f"{w} ({k:,})" for w, k in tok.most_common(top_tokens)), "",
          "## Most frequent first words", "", ", ".join(f"{w} ({k:,})" for w, k in first.most_common(25))]
    stats["top_tokens"] = tok.most_common(top_tokens)
    stats["distinct_tokens"] = len(tok)

    # ---- examples
    def bullets(idx: list[int]) -> list[str]:
        return [f"- `{i}`: {cleaned[i]}" for i in idx]

    L += ["", f"## Random sample ({sample_size}, seed {seed})", ""] + bullets(rng.sample(range(n), min(sample_size, n)))
    order = sorted(range(n), key=lambda i: (words[i], i))
    L += ["", "## Shortest phrases", ""] + bullets(order[:15]) + ["", "## Longest phrases", ""] + bullets(order[-10:][::-1])
    L += ["", "## Examples per category", ""]
    for k, (desc, _) in CATEGORIES.items():
        if hits[k]:
            L += [f"**{k}** ({desc})", ""] + bullets(rng.sample(hits[k], min(per_cat, len(hits[k])))) + [""]
    if multiline:
        L += ["## Concepts with an embedded line break (kept verbatim in the bank)", ""] + [f"- `{i}`: {texts[i]!r}" for i in multiline[:5]]
    return "\n".join(L) + "\n", stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="profile YAML overlaid on configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--grep", default=None, help="regex to search concepts for (case-insensitive) instead of the full report")
    ap.add_argument("--max", type=int, default=25, help="with --grep: how many matches to print (first N)")
    ap.add_argument("--sample", type=int, default=0, help="with --grep: print a random sample of N matches instead of the first N")
    ap.add_argument("--no-write", action="store_true", help="do not write the report files")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    csv_path = require_file(s.paths.concepts_csv, "paths.concepts_csv")
    texts = load_concept_texts(csv_path, s.concept_bank.expected_n_concepts)

    if args.grep:
        rx = re.compile(args.grep, re.IGNORECASE)
        matches = [i for i, t in enumerate(texts) if rx.search(clean_text(t))]
        print(f"{len(matches):,} of {len(texts):,} concepts match /{args.grep}/ ({100 * len(matches) / len(texts):.2f}%)")
        show = random.Random(s.explore.seed).sample(matches, min(args.sample, len(matches))) if args.sample else matches[: args.max]
        for i in show:
            print(f"{i:>7}  {clean_text(texts[i])}")
        return

    md, stats = build_report(texts, s.explore.seed, s.explore.sample_size, s.explore.top_tokens, s.explore.examples_per_category)
    print(md)
    if not args.no_write:
        out = s.paths.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "concept_exploration.md").write_text(md, encoding="utf-8")
        (out / "concept_exploration.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[written] {out / 'concept_exploration.md'}  and  concept_exploration.json", file=sys.stderr)


if __name__ == "__main__":
    main()
