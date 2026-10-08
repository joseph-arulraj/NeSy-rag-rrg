"""Fix the knowledge-rule choices by user decision (2026-10-07): copy a rules_eval run's region thresholds and
measured R1/R2 numbers, and set keep flags explicitly (R2 side agreement kept, R1 localisation support dropped).
Writes rules_eval.json in this run dir in the same format, for Stage 8 and the case evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--r1", choices=["keep", "drop"], required=True)
    ap.add_argument("--r2", choices=["keep", "drop"], required=True)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("rules-set", vars(a), a.run_dir) as run:
        r = json.loads((Path(a.from_run) / "rules_eval.json").read_text())
        measured = {k: r[k]["keep"] for k in ("R1", "R2")}
        r["R1"]["keep"], r["R2"]["keep"] = a.r1 == "keep", a.r2 == "keep"
        r["decision"] = {"source": "user decision 2026-10-07", "measured_keep_by_CI_rule": measured,
                         "R1": a.r1, "R2": a.r2}
        (run.dir / "rules_eval.json").write_text(json.dumps(r, indent=2, default=float))
        run.log(f"rules fixed by user decision: R1 {a.r1}, R2 {a.r2} (CI rule on this head said {measured})")


if __name__ == "__main__":
    main()
