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
from typing import Callable, Optional

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
    """Two distinct problems, both confirmed on real pipeline output, not hypothetical:

    (1) B (and C) can each independently produce a no_finding node, and B can further fragment
        it by anatomy (the same fragmentation pattern seen elsewhere, e.g. pleural_effusion) --
        left unmerged, this renders as several redundant "no evidence of abnormality" sentences.
        No_finding is ALWAYS deduplicated to its single highest-confidence node first, regardless
        of what else happens below.

    (2) The original design compared no_finding's score against only the SINGLE strongest other
        finding, one-on-one -- so one high-confidence no_finding node could veto every other
        admitted finding at once, each in an individually-"won" pairwise comparison. Confirmed on
        real output: no_finding at 0.995 beat 8 separate independently-admitted findings one at a
        time, including two others independently above 0.9 confidence, wiping out all of them.
        That is not defensible: no_finding's whole claim is "nothing ELSE is going on", which
        becomes implausible once there is more than one independent finding backing the opposite,
        regardless of what no_finding's own single calibrated score happens to be (itself possibly
        just a calibration artifact given how sparse the validation data is for this label -- see
        the calibration-quality discussion earlier). no_finding now only wins when there is AT
        MOST ONE competing finding and it's weaker."""
    no_finding_nodes = [f for f in findings if f.label == "no_finding" and f.polarity == "present"]
    if not no_finding_nodes:
        return findings, [], []

    audit: list[RuleApplication] = []
    suppressed: list[Finding] = []
    kept = list(findings)
    best_no_finding = max(no_finding_nodes, key=lambda f: f.confidence)

    extra_no_finding = [f for f in no_finding_nodes if f is not best_no_finding]
    if extra_no_finding:
        for f in extra_no_finding:
            audit.append(RuleApplication(
                rule_id="mutual_exclusivity_no_finding", rule_version="1",
                targets=[best_no_finding.finding_id, f.finding_id], action="drop",
                before={"polarity": f.polarity, "confidence": f.confidence}, after={"polarity": "suppressed"},
                rationale=(f"redundant 'no_finding' node (confidence {f.confidence:.3f}) -- "
                           f"'{best_no_finding.finding_id}' (confidence {best_no_finding.confidence:.3f}) already covers this"),
            ))
        suppressed.extend(extra_no_finding)
        kept = [f for f in kept if f not in extra_no_finding]

    other_present = [f for f in kept if f.label != "no_finding" and f.polarity == "present"]
    if not other_present:
        return kept, audit, suppressed

    if len(other_present) == 1 and best_no_finding.confidence >= other_present[0].confidence:
        f = other_present[0]
        audit.append(RuleApplication(
            rule_id="mutual_exclusivity_no_finding", rule_version="1",
            targets=[best_no_finding.finding_id, f.finding_id], action="drop",
            before={"polarity": f.polarity, "confidence": f.confidence}, after={"polarity": "suppressed"},
            rationale=(f"'no_finding' (confidence {best_no_finding.confidence:.3f}) outranks the only "
                       f"co-asserted finding '{f.label}' (confidence {f.confidence:.3f})"),
        ))
        suppressed.append(f)
        kept = [x for x in kept if x is not f]
        return kept, audit, suppressed

    audit.append(RuleApplication(
        rule_id="mutual_exclusivity_no_finding", rule_version="1",
        targets=[best_no_finding.finding_id] + [f.finding_id for f in other_present],
        action="drop", before={"polarity": best_no_finding.polarity, "confidence": best_no_finding.confidence},
        after={"polarity": "suppressed"},
        rationale=(f"{len(other_present)} independently co-asserted finding(s) outweigh 'no_finding' "
                   f"(confidence {best_no_finding.confidence:.3f}) taken together, regardless of its own score"),
    ))
    suppressed.append(best_no_finding)
    kept = [x for x in kept if x is not best_no_finding]
    return kept, audit, suppressed


def _suppress_negative_no_finding(findings: list[Finding], rules: RulesTable):
    """'no_finding' is a meta-claim ("nothing else is wrong"), not an ordinary pathology --
    unlike every other label, a NEGATIVE assertion of it is a double negative ("it is false that
    there's no finding" = "something IS wrong") that renders as confusing, backwards-sounding
    prose ("There is no evidence of abnormality") sitting right next to a list of real findings
    that says the opposite (confirmed directly in real pipeline output, not hypothetical). An
    absent/uncertain no_finding node adds no information the other admitted findings don't
    already convey, so it's dropped outright rather than rendered."""
    targets = [f for f in findings if f.label == "no_finding" and f.polarity != "present"]
    if not targets:
        return findings, [], []
    audit = [
        RuleApplication(
            rule_id="suppress_negative_no_finding", rule_version="1", targets=[f.finding_id],
            action="drop", before={"polarity": f.polarity, "confidence": f.confidence}, after={"polarity": "suppressed"},
            rationale="'no_finding' asserted as absent/uncertain is a confusing double-negative and adds no "
                      "information beyond the other admitted findings -- dropped rather than rendered",
        )
        for f in targets
    ]
    return [f for f in findings if f not in targets], audit, targets


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


_PASSES = [_attribute_validity, _suppress_negative_no_finding, _mutual_exclusivity_no_finding, _laterality_consistency]


def _fingerprint(findings: list[Finding]) -> frozenset:
    """A cheap, order-independent signature of the graph's state -- used to detect TRUE
    non-convergence (the exact same state recurring) separately from "still making progress,
    just needs more rounds" (a large, real graph can legitimately take a few iterations to
    settle when rules interact, e.g. attribute_validity freeing up a node that
    laterality_consistency then needs to look at)."""
    return frozenset((f.finding_id, f.label, f.laterality.value, f.polarity, round(f.confidence, 6)) for f in findings)


# --------------------------------------------------------------------------- orchestration
def apply_rules(graph: BeliefGraph, settings: Settings, log: Optional[Callable[[str], None]] = None) -> BeliefGraph:
    rules = load_rules_table(settings.rules.rules_path)
    findings = list(graph.findings)
    all_audit: list[RuleApplication] = []
    all_suppressed: list[Finding] = list(graph.suppressed)
    seen_fingerprints: dict[frozenset, int] = {}

    for iteration in range(settings.rules.max_cascade_iterations):
        changed = False
        iter_summary = []
        for rule_pass in _PASSES:
            findings, audit, suppressed = rule_pass(findings, rules)
            if audit:
                changed = True
                all_audit.extend(audit)
                all_suppressed.extend(suppressed)
                iter_summary.append(f"{rule_pass.__name__}: {len(audit)} action(s) ({[a.rule_id for a in audit][:3]}...)")
        if log:
            log(f"    N25 iteration {iteration + 1}: " + ("; ".join(iter_summary) if iter_summary else "no changes"))
        if not changed:
            break
        fp = _fingerprint(findings)
        if fp in seen_fingerprints:
            raise RuleEngineCascadeError(
                f"N25 rules are oscillating, not converging: the exact same graph state recurred "
                f"at iteration {seen_fingerprints[fp] + 1} and again at {iteration + 1} "
                f"(rules.max_cascade_iterations={settings.rules.max_cascade_iterations}). "
                f"This is a real cycle in the rule logic, not just a slow-but-correct cascade -- "
                f"check which rule_id keeps re-firing on the same finding_ids in the per-iteration log above."
            )
        seen_fingerprints[fp] = iteration
    else:
        raise RuleEngineCascadeError(
            f"N25 rule application was still changing every iteration after "
            f"rules.max_cascade_iterations={settings.rules.max_cascade_iterations} rounds, without repeating a "
            f"prior state -- may just need a higher limit for this study's finding count "
            f"({len(graph.findings)} initial findings), or may be a slow-converging real cycle. "
            f"Check the per-iteration log above for which rule kept firing."
        )

    return BeliefGraph(
        findings=findings, relations=graph.relations, study_meta=graph.study_meta,
        graph_version="verified", audit=all_audit, rejected=graph.rejected, suppressed=all_suppressed,
    )
