"""Load and validate the knowledge-graph YAML with plain Python (no triple store, no Prolog).

Two files: kg/findings.yaml (findings, is_a, may_occur_in, associated_with, phrasing, pertinent
negatives) and kg/anatomy.yaml (anatomy nodes with part_of, generated from RadLex). Both are checked
on load: required keys, unique ids, every reference resolves, is_a and part_of are acyclic. A
malformed file raises KGError naming the entry.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

KG_DIR = Path(__file__).resolve().parents[1] / "kg"
SIDES = {"left", "right", "midline", "bilateral"}
PHRASE_KEYS = {"present", "absent", "possible"}


class KGError(ValueError):
    pass


def _need(entry: dict, keys, where: str):
    for k in keys:
        if k not in entry or entry[k] in (None, ""):
            raise KGError(f"{where}: entry {entry.get('id', entry)} lacks required key '{k}'")


def _acyclic(edges: dict[str, list[str]], nodes, what: str):
    state: dict[str, int] = {}

    def visit(n, path):
        if state.get(n) == 1:
            raise KGError(f"{what} cycle: {' -> '.join(path + [n])}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for p in edges.get(n, []):
            visit(p, path + [n])
        state[n] = 2
    for n in nodes:
        visit(n, [])


def validate_anatomy(d: dict) -> dict:
    if not isinstance(d, dict) or "nodes" not in d:
        raise KGError("anatomy: missing 'nodes'")
    ids = []
    for n in d["nodes"]:
        _need(n, ("id", "label", "side"), "anatomy")
        if n["side"] not in SIDES:
            raise KGError(f"anatomy: node {n['id']} has invalid side {n['side']}")
        ids.append(n["id"])
    if len(set(ids)) != len(ids):
        raise KGError("anatomy: duplicate node id")
    parent = {}
    for n in d["nodes"]:
        if n.get("part_of"):
            if n["part_of"] not in ids:
                raise KGError(f"anatomy: {n['id']} part_of unknown node {n['part_of']}")
            parent[n["id"]] = [n["part_of"]]
    _acyclic(parent, ids, "part_of")
    for region, node in d.get("imagenome_region_to_node", {}).items():
        if node not in ids:
            raise KGError(f"anatomy: region {region!r} maps to unknown node {node}")
    d["_ids"] = ids
    d["_parent"] = {k: v[0] for k, v in parent.items()}
    d["_node"] = {n["id"]: n for n in d["nodes"]}
    return d


def validate_findings(d: dict, anatomy_ids) -> dict:
    if not isinstance(d, dict) or "findings" not in d:
        raise KGError("findings: missing 'findings'")
    ids = []
    for f in d["findings"]:
        _need(f, ("id", "label", "lateralisable", "definition", "phrase"), "findings")
        if set(f["phrase"]) != PHRASE_KEYS:
            raise KGError(f"findings: {f['id']} phrase keys must be {sorted(PHRASE_KEYS)}")
        ids.append(f["id"])
    if len(set(ids)) != len(ids):
        raise KGError("findings: duplicate finding id")
    seen = set()

    def edge_ids(section, keys):
        for e in d.get(section, []):
            _need(e, ("id", "rationale", *keys), section)
            if e["id"] in seen:
                raise KGError(f"{section}: duplicate id {e['id']}")
            seen.add(e["id"])
            yield e
    parents: dict[str, list[str]] = {}
    for e in edge_ids("is_a", ("child", "parent")):
        for k in ("child", "parent"):
            if e[k] not in ids:
                raise KGError(f"is_a {e['id']}: unknown finding {e[k]}")
        if e["child"] == e["parent"]:
            raise KGError(f"is_a {e['id']}: self loop")
        parents.setdefault(e["child"], []).append(e["parent"])
    _acyclic(parents, ids, "is_a")
    moi = {}
    for e in edge_ids("may_occur_in", ("finding", "anatomy")):
        if e["finding"] not in ids:
            raise KGError(f"may_occur_in {e['id']}: unknown finding {e['finding']}")
        for a in e["anatomy"]:
            if a not in anatomy_ids:
                raise KGError(f"may_occur_in {e['id']}: unknown anatomy {a}")
        moi[e["finding"]] = e
    for e in edge_ids("associated_with", ("a", "b")):
        for k in ("a", "b"):
            if e[k] not in ids:
                raise KGError(f"associated_with {e['id']}: unknown finding {e[k]}")
    for e in edge_ids("pertinent_negatives", ("finding", "when")):
        if e["finding"] not in ids:
            raise KGError(f"pertinent_negatives {e['id']}: unknown finding {e['finding']}")
        w = e["when"]
        if not (w == "always" or (isinstance(w, dict) and set(w) == {"any_present"} and set(w["any_present"]) <= set(ids))):
            raise KGError(f"pertinent_negatives {e['id']}: 'when' must be 'always' or {{any_present: [findings]}}")
    nc = d.get("normal_call")
    if nc is not None:
        _need(nc, ("id", "status", "critical", "rationale"), "normal_call")
        for lst in [nc["critical"], *nc.get("candidates", {}).values()]:
            for f in lst:
                if f not in ids:
                    raise KGError(f"normal_call: unknown finding {f}")
    d["_ids"] = ids
    d["_parents"] = parents
    d["_moi"] = moi
    d["_by_id"] = {f["id"]: f for f in d["findings"]}
    return d


MATCHES = {"exact", "approximate", "none"}


def validate_radlex(d: dict, finding_ids, anatomy_ids, device_ids) -> dict:
    """kg/radlex_map.yaml: every entry names a known finding / anatomy node / device (zones are free keys),
    has a match in {exact, approximate, none}, a RadLex id iff match != none, and a source."""
    if "entries" not in d:
        raise KGError("radlex_map: missing 'entries'")
    known = {"finding": set(finding_ids), "anatomy": set(anatomy_ids), "device": set(device_ids)}
    seen = set()
    for e in d["entries"]:
        _need(e, ("kind", "key", "match", "source"), "radlex_map")
        if e["match"] not in MATCHES:
            raise KGError(f"radlex_map: {e['key']} has invalid match {e['match']}")
        if (e["match"] == "none") != (e.get("radlex_id") is None):
            raise KGError(f"radlex_map: {e['key']}: radlex_id must be present iff match != none")
        if e.get("radlex_id") and not str(e["radlex_id"]).startswith("RID"):
            raise KGError(f"radlex_map: {e['key']}: bad RadLex id {e['radlex_id']}")
        if e["kind"] in known and e["key"] not in known[e["kind"]]:
            raise KGError(f"radlex_map: unknown {e['kind']} {e['key']}")
        if (e["kind"], e["key"]) in seen:
            raise KGError(f"radlex_map: duplicate entry {e['kind']} {e['key']}")
        seen.add((e["kind"], e["key"]))
    missing = [f for f in finding_ids if ("finding", f) not in seen]
    if missing:
        raise KGError(f"radlex_map: findings without an entry: {missing}")
    return d


@lru_cache(maxsize=None)
def load_devices(path: str | Path = KG_DIR / "devices.yaml") -> dict:
    d = yaml.safe_load(Path(path).read_text())
    for e in d.get("device_types", []):
        _need(e, ("id", "finding", "match", "source"), "devices")
        if e["finding"] not in load_findings()["_by_id"]:
            raise KGError(f"devices: {e['id']} maps to unknown finding {e['finding']}")
    return d


@lru_cache(maxsize=None)
def load_radlex(path: str | Path = KG_DIR / "radlex_map.yaml") -> dict:
    d = yaml.safe_load(Path(path).read_text())
    return validate_radlex(d, finding_ids(), [n["id"] for n in load_anatomy()["nodes"]],
                           [e["id"] for e in load_devices().get("device_types", [])])


@lru_cache(maxsize=None)
def load_anatomy(path: str | Path = KG_DIR / "anatomy.yaml") -> dict:
    return validate_anatomy(yaml.safe_load(Path(path).read_text()))


@lru_cache(maxsize=None)
def load_findings(path: str | Path = KG_DIR / "findings.yaml", anatomy_path: str | Path = KG_DIR / "anatomy.yaml") -> dict:
    return validate_findings(yaml.safe_load(Path(path).read_text()), set(load_anatomy(anatomy_path)["_ids"]))


def finding_ids() -> list[str]:
    return list(load_findings()["_ids"])


def parents_of(f: str) -> list[str]:
    return list(load_findings()["_parents"].get(f, []))


def children_of(f: str) -> list[str]:
    return [c for c, ps in load_findings()["_parents"].items() if f in ps]


def ancestors(f: str) -> list[str]:
    out, todo = [], parents_of(f)
    while todo:
        p = todo.pop()
        if p not in out:
            out.append(p)
            todo += parents_of(p)
    return out


def descendants(f: str) -> list[str]:
    out, todo = [], children_of(f)
    while todo:
        c = todo.pop()
        if c not in out:
            out.append(c)
            todo += children_of(c)
    return out


def topo_order() -> list[str]:
    """Children before parents."""
    ids = finding_ids()
    depth = {f: len(ancestors(f)) for f in ids}
    return sorted(ids, key=lambda f: -depth[f])


def anatomy_path(node: str) -> list[str]:
    """node, its part_of parent, grandparent, ... up to the root."""
    a = load_anatomy()
    out = [node]
    while out[-1] in a["_parent"]:
        out.append(a["_parent"][out[-1]])
    return out
