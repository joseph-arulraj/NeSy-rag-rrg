"""Four-band thresholds. v2 (user decision 2026-10-06, later the same day): present and possible are
fixed CALIBRATED-PROBABILITY cut-offs, no longer precision-derived thresholds. The PPV rule had placed
effusion "present" at p = 0.46 and "possible lung opacity" at p = 0.21.

  present   p >= 0.70 (a finding whose probability never reaches 0.70 never enters "present")
  possible  0.40 <= p < 0.70
  absent    highest threshold missing <= 5 % of positives (pneumothorax <= 2 %)
  silent    everything else
An unreachable tier is None, and the finding never enters it.
Overlap rule (2026-10-06): for common findings (prevalence > 0.4) "PPV >= 0.4" is met by flagging every
study, so the possible threshold can fall below the absent threshold. The absent rule then takes
priority: a study is absent if p <= absent, and possible only covers (absent, present). The absent
threshold is dropped only if it reaches the present threshold.
Non-reportable nodes (the any_abnormality root) are never stated, so only their absent tier is fitted;
their band is otherwise lifted by their children through D2.
"""
from __future__ import annotations

import numpy as np

from .evaluate import threshold_candidates

PRESENT_P, POSSIBLE_P = 0.70, 0.40        # calibrated-probability cut-offs (v2)
PRESENT_PPV, POSSIBLE_PPV = 0.7, 0.4      # v1 precision targets, kept only for the reported stats
MISS = {"pneumothorax": 0.02, "any_abnormality": 0.02}   # root at 2 % (user decision 2026-10-06)
DEFAULT_MISS = 0.05


def fit(y: np.ndarray, p: np.ndarray, finding: str) -> dict:
    c = {(r["band"], r["rule"]): r for r in threshold_candidates(y, p, miss_targets=(MISS.get(finding, DEFAULT_MISS),),
                                                                 ppv_targets=(POSSIBLE_PPV, PRESENT_PPV))}

    def thr(key):
        r = c.get(key)
        return None if r is None or r.get("thr") is None or np.isnan(r.get("thr", np.nan)) else float(r["thr"])
    present, possible = PRESENT_P, POSSIBLE_P
    from . import kg
    if not kg.load_findings()["_by_id"][finding].get("reportable", True):
        present = possible = None
    if possible is not None and present is not None and possible >= present:
        possible = None
    absent = thr(("absent", f"missed_pos<={MISS.get(finding, DEFAULT_MISS)}"))
    if absent is not None and present is not None and absent >= present:
        absent = None
    overlap = possible is not None and absent is not None and possible <= absent
    out = {"present": present, "possible": possible, "absent": absent,
           "rule": f"present p>={PRESENT_P}, possible p>={POSSIBLE_P}, absent missed<={MISS.get(finding, DEFAULT_MISS)}",
           "absent_overrides_possible": bool(overlap)}
    return out
