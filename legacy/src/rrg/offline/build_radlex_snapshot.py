"""Parse a downloaded RadLex.owl once into a compact local snapshot (pipeline.md 5.4 rev 2:
no live BioPortal calls at runtime, per INV-4). Pure stdlib XML -- no rdflib/reasoner needed;
RadLex's logical content beyond subclass + existential part-of restrictions was checked and
found not to need one (this project's RadLex investigation).

Extracts, per class: English preferred label, synonyms, direct is-a parents (rdfs:subClassOf),
and the Part_Of-family closure (Part_Of, Regional_Part_Of, Constitutional_Part_Of,
Branch_Part_of, Segment_Of, and their Has_* inverses -- validated against RadLex's own OWL
object-property axioms, not hand-picked). Retired/obsolete stub classes (label == bare RID)
are dropped.

  python offline/build_radlex_snapshot.py --config configs/hpc.yaml
"""
from __future__ import annotations

import gzip
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from ..core.config import Settings, require_file
from ..core.errors import PipelineError

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
XML = "{http://www.w3.org/XML/1998/namespace}"
RID_NS = "http://www.radlex.org/RID/"

# Sub-properties of Part_Of and their Has_* inverses, per RadLex's own declared OWL object-property
# axioms (Regional_Part_Of/Constitutional_Part_Of/Branch_Part_of/Segment_Of are all sub-properties of
# Part_Of; Has_Part is Part_Of's inverse, etc.) -- NOT a hand-picked list, confirmed against the file.
PART_OF_FAMILY = {"Part_Of", "Regional_Part_Of", "Constitutional_Part_Of", "Branch_Part_of", "Segment_Of"}
HAS_PART_FAMILY = {"Has_Part", "Has_Regional_Part", "Has_Constitutional_Part", "Has_Branch_Part"}
_STUB_LABEL = re.compile(r"^RID\d+$")


class RadLexParseError(PipelineError):
    code = "E_RADLEX_PARSE"


@dataclass
class _RawClass:
    rid: str
    labels: dict[str, str]            # lang -> text
    synonyms: set[str]
    is_a_parents: set[str]
    part_of_targets: set[str]         # this class IS part of these (forward)
    has_part_targets: set[str]        # this class HAS these as parts (inverse direction)
    anatomical_site: set[str]


def _local(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def parse_radlex_owl(owl_path: Path, log=print) -> dict[str, _RawClass]:
    classes: dict[str, _RawClass] = defaultdict(
        lambda: _RawClass("", {}, set(), set(), set(), set(), set())
    )

    def get(rid: str) -> _RawClass:
        c = classes[rid]
        if not c.rid:
            c.rid = rid
        return c

    n = 0
    t0 = time.time()
    for _, el in ET.iterparse(str(owl_path)):
        tag = el.tag
        if tag == OWL + "Class":
            about = el.get(RDF + "about")
            if not about or "/RID/" not in about:
                el.clear()
                continue
            rid = _local(about)
            c = get(rid)
            for ch in el:
                if ch.tag == RDFS + "label":
                    lang = ch.get(XML + "lang") or "?"
                    if ch.text:
                        c.labels[lang] = ch.text
                elif ch.tag == RID_NS.replace("http://www.radlex.org/RID/", "{" + RID_NS + "}") + "Synonym" or (
                    ch.tag.startswith("{" + RID_NS + "}") and ch.tag.endswith("}Synonym")
                ):
                    if ch.text:
                        c.synonyms.add(ch.text)
                elif ch.tag == RDFS + "subClassOf":
                    target = ch.get(RDF + "resource")
                    if target and "/RID/" in target:
                        c.is_a_parents.add(_local(target))
                        continue
                    restr = ch.find(OWL + "Restriction")
                    if restr is None:
                        continue
                    prop_el = restr.find(OWL + "onProperty")
                    val_el = restr.find(OWL + "someValuesFrom")
                    if prop_el is None or val_el is None:
                        continue
                    prop = _local(prop_el.get(RDF + "resource", ""))
                    val = val_el.get(RDF + "resource")
                    if not val or "/RID/" not in val:
                        continue
                    val_rid = _local(val)
                    if prop in PART_OF_FAMILY:
                        c.part_of_targets.add(val_rid)
                    elif prop in HAS_PART_FAMILY:
                        c.has_part_targets.add(val_rid)
                    elif prop == "Anatomical_Site":
                        c.anatomical_site.add(val_rid)
            n += 1
            if n % 20000 == 0:
                log(f"  ...{n:,} owl:Class elements ({time.time() - t0:.0f}s)")
            el.clear()
        elif tag == RDF + "Description":
            about = el.get(RDF + "about")
            if not about or "/RID/" not in about:
                el.clear()
                continue
            rid = _local(about)
            c = get(rid)
            for ch in el:
                if ch.tag == RDFS + "label":
                    lang = ch.get(XML + "lang") or "?"
                    if ch.text:
                        c.labels[lang] = ch.text
                    continue
                local = ch.tag.rsplit("}", 1)[-1] if "}" in ch.tag else ch.tag
                if local == "Synonym" and ch.text:
                    c.synonyms.add(ch.text)
                elif local in PART_OF_FAMILY:
                    target = ch.get(RDF + "resource")
                    if target and "/RID/" in target:
                        c.part_of_targets.add(_local(target))
                elif local in HAS_PART_FAMILY:
                    target = ch.get(RDF + "resource")
                    if target and "/RID/" in target:
                        c.has_part_targets.add(_local(target))
                elif local == "Anatomical_Site":
                    target = ch.get(RDF + "resource")
                    if target and "/RID/" in target:
                        c.anatomical_site.add(_local(target))
            el.clear()
    log(f"parsed {n:,} classes in {time.time() - t0:.0f}s")
    return dict(classes)


def build_snapshot(owl_path: Path, version: str, log=print) -> dict:
    raw = parse_radlex_owl(owl_path, log=log)

    live: dict[str, dict] = {}
    for rid, c in raw.items():
        label = c.labels.get("en")
        if not label or _STUB_LABEL.match(label):
            continue   # retired stub, or no English label at all
        live[rid] = {
            "label": label,
            "synonyms": sorted(c.synonyms),
            "is_a_parents": sorted(c.is_a_parents),
            "part_of_targets": sorted(c.part_of_targets),
            "has_part_targets": sorted(c.has_part_targets),
            "anatomical_site": sorted(c.anatomical_site),
        }

    # Kept as TWO directed lists, not merged: part_of ("this class is part of these containers")
    # vs has_part ("this class contains these parts"). A closure walk must descend
    # container -> part only (never ascend through a container to a sibling branch) -- merging
    # them into one symmetric relation was tried and produced a 27,685-class "chest" scope by
    # leaking through thorax's own containers into unrelated body regions. Keeping them directed
    # lets RadLexClient build a correct descend-only graph for scoping, while a class's DIRECT
    # union of the two (for pairwise "are A and B adjacent" checks, not scoping) is still cheap
    # to compute from these two lists at load time.
    for rid, entry in live.items():
        entry["part_of"] = entry.pop("part_of_targets")
        entry["has_part"] = entry.pop("has_part_targets")

    log(f"live (non-retired, English-labelled) classes: {len(live):,} of {len(raw):,} total")
    return {"version": version, "classes": live}


def main(settings: Settings, log=print) -> Path:
    rl = settings.radlex
    owl_path = require_file(rl.raw_owl_path, "radlex.raw_owl_path")
    snapshot = build_snapshot(owl_path, rl.version, log=log)

    out_path = Path(rl.snapshot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(snapshot, fh, separators=(",", ":"))
    tmp.replace(out_path)
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rrg.core.config import load_settings  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    main(load_settings(args.config, args.overrides))
