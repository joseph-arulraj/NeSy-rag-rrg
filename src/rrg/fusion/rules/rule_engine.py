"""N25 `KG + medical rule check` (pipeline.md §5.4). Symbolic validation over the initial belief
graph: removes or down-weights implausible/conflicting claims, non-destructively (suppressed
findings are kept, never deleted -- the "non-destructive" requirement) with a full
`RuleApplication` audit trail per action (deterministic ordering -- INV-4).

Implements three of pipeline.md's rule categories concretely: attribute validity (a laterality
value outside a finding's valid range), mutual exclusivity (no_finding vs. any other asserted
finding), and laterality consistency (the same finding/anatomy asserted both left and right by
different sources) -- the margin-based resolution policy worked out in this project's N25 design
discussion: a clear confidence margin trusts the winner outright; a narrow margin widens to
`unspecified` rather than guess, with both contested sides recorded in the audit.

NOT implemented here -- flagged, not silently skipped:
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from ...core.config import Settings
from ...core.errors import RuleEngineCascadeError
from ...core.types import BeliefGraph, Finding, Laterality, RuleApplication

KNOWN_GAPS = [
    "hierarchy/subsumption (a specific finding and its parent term both asserted) -- not implemented",
    "device/context plausibility (finding requires a device or prior state not otherwise supported) -- not implemented",
    "RadLex-based anatomical validity (finding asserted at an anatomically impossible site) -- not "
    "implemented; only the hand-curated attribute-validity laterality check below is live",
]


@dataclass(frozen=True)
class RulesTable:
    non_lateralizable_findings: frozenset
    mutual_exclusive_with_no_finding: bool
    laterality_margin_threshold: float


def load_rules_table(path: Path) -> RulesTable:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return RulesTable(
        non_lateralizable_findings=frozenset(data.get("non_lateralizable_findings", [])),
        mutual_exclusive_with_no_finding=bool(data.get("mutual_exclusive_with_no_finding", True)),
        laterality_margin_threshold=float(data.get("laterality_margin_threshold", 0.1)),
    )


# --------------------------------------------------------------------------- rule passes
def _attribute_validity(findings: list[Finding], rules: RulesTable):
    """'attribute value outside the permitted range for that finding -> drop attribute, keep
    finding' (pipeline.md §5.4's rule table)."""
    out, audit, suppressed = [], [], []
    for f in findings:
        if f.label in rules.non_lateralizable_findings and f.laterality not in (Laterality.UNSPECIFIED, Laterality.MIDLINE):
            before = {"laterality": f.laterality.value}
            audit.append(RuleApplication(
                rule_id="attribute_validity_laterality", rule_version="1", targets=[f.finding_id],
                action="drop_attribute", before=before, after={"laterality": "unspecified"},
                rationale=f"{f.label} is a whole-organ/midline finding class; laterality is not a valid attribute for it",
            ))
            out.append(replace(f, laterality=Laterality.UNSPECIFIED))
        else:
            out.append(f)
    return out, audit, suppressed


def _mutual_exclusivity_no_finding(findings: list[Finding], rules: RulesTable):
    if not rules.mutual_exclusive_with_no_finding:
        return findings, [], []
    no_finding_nodes = [f for f in findings if f.label == "no_finding" and f.polarity == "present"]
    other_present = [f for f in findings if f.label != "no_finding" and f.polarity == "present"]
    if not no_finding_nodes or not other_present:
        return findings, [], []

    audit = []
    best_no_finding = max(no_finding_nodes, key=lambda f: f.confidence)
    best_other = max(other_present, key=lambda f: f.confidence)
    if best_no_finding.confidence >= best_other.confidence:
        losers = other_present
        for f in losers:
            audit.append(RuleApplication(
                rule_id="mutual_exclusivity_no_finding", rule_version="1",
                targets=[best_no_finding.finding_id, f.finding_id], action="drop",
                before={"polarity": f.polarity, "confidence": f.confidence}, after={"polarity": "suppressed"},
                rationale=(f"'no_finding' (confidence {best_no_finding.confidence:.3f}) outranks "
                           f"co-asserted '{f.label}' (confidence {f.confidence:.3f})"),
            ))
        return [f for f in findings if f not in losers], audit, losers

    audit.append(RuleApplication(
        rule_id="mutual_exclusivity_no_finding", rule_version="1", targets=[best_no_finding.finding_id],
        action="drop", before={"polarity": best_no_finding.polarity, "confidence": best_no_finding.confidence},
        after={"polarity": "suppressed"},
        rationale=(f"co-asserted finding '{best_other.label}' (confidence {best_other.confidence:.3f}) "
                   f"outranks 'no_finding' (confidence {best_no_finding.confidence:.3f})"),
    ))
    return [f for f in findings if f is not best_no_finding], audit, [best_no_finding]


def _laterality_consistency(findings: list[Finding], rules: RulesTable):
    """Same finding/anatomy asserted both left-only and right-only by different sources ->
    resolve by confidence margin, or widen to `unspecified` if no side has a confident lead
    (this project's N25 design discussion; see docs/pipeline.md AMBIG-7)."""
    by_key: dict[tuple, list[Finding]] = {}
    for f in findings:
        if f.laterality in (Laterality.LEFT, Laterality.RIGHT) and f.polarity == "present":
            by_key.setdefault((f.label, f.anatomy), []).append(f)

    audit, suppressed = [], []
    kept = list(findings)
    for (label, _anatomy), nodes in by_key.items():
        if len({n.laterality for n in nodes}) < 2:
            continue  # fewer than 2 nodes, or both on the same side -- not a conflict
        nodes_sorted = sorted(nodes, key=lambda n: -n.confidence)
        winner, runner_up = nodes_sorted[0], nodes_sorted[1]
        margin = winner.confidence - runner_up.confidence

        if margin >= rules.laterality_margin_threshold:
            losers = nodes_sorted[1:]
            for f in losers:
                audit.append(RuleApplication(
                    rule_id="laterality_consistency", rule_version="1", targets=[winner.finding_id, f.finding_id],
                    action="drop", before={"laterality": f.laterality.value, "confidence": f.confidence},
                    after={"laterality": "suppressed"},
                    rationale=(f"'{label}' laterality conflict: {winner.laterality.value} "
                               f"(confidence {winner.confidence:.3f}) outranks {f.laterality.value} "
                               f"(confidence {f.confidence:.3f}) by margin {margin:.3f} >= "
                               f"{rules.laterality_margin_threshold}"),
                ))
            suppressed.extend(losers)
            kept = [f for f in kept if f not in losers]
        else:
            for f in nodes_sorted:
                audit.append(RuleApplication(
                    rule_id="laterality_consistency", rule_version="1",
                    targets=[n.finding_id for n in nodes_sorted], action="resolve",
                    before={"laterality": f.laterality.value, "confidence": f.confidence},
                    after={"laterality": "unspecified"},
                    rationale=(f"'{label}' laterality conflict has no confident winner (margin "
                               f"{margin:.3f} < {rules.laterality_margin_threshold}); widened to "
                               f"unspecified rather than guessed"),
                ))
            suppressed.extend(nodes_sorted)
            kept = [f for f in kept if f not in nodes_sorted]
            kept.append(replace(
                winner, finding_id=f"{winner.finding_id}+merged", laterality=Laterality.UNSPECIFIED,
                confidence=max(n.confidence for n in nodes_sorted),
            ))
    return kept, audit, suppressed


_PASSES = [_attribute_validity, _mutual_exclusivity_no_finding, _laterality_consistency]


# --------------------------------------------------------------------------- orchestration
def apply_rules(graph: BeliefGraph, settings: Settings) -> BeliefGraph:
    rules = load_rules_table(settings.rules.rules_path)
    findings = list(graph.findings)
    all_audit: list[RuleApplication] = []
    all_suppressed: list[Finding] = list(graph.suppressed)

    for _iteration in range(settings.rules.max_cascade_iterations):
        changed = False
        for rule_pass in _PASSES:
            findings, audit, suppressed = rule_pass(findings, rules)
            if audit:
                changed = True
                all_audit.extend(audit)
                all_suppressed.extend(suppressed)
        if not changed:
            break
    else:
        raise RuleEngineCascadeError(
            f"N25 rule application did not converge within rules.max_cascade_iterations={settings.rules.max_cascade_iterations}"
        )

    return BeliefGraph(
        findings=findings, relations=graph.relations, study_meta=graph.study_meta,
        graph_version="verified", audit=all_audit, rejected=graph.rejected, suppressed=all_suppressed,
    )
