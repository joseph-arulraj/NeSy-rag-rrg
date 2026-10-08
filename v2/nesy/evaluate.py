"""Metrics. Test-split metrics are never computed by these helpers' callers until the end."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    out = {"n": int(len(y)), "pos": int(y.sum())}
    if 0 < y.sum() < len(y):
        out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                   ece=ece(y, p), brier=float(np.mean((p - y) ** 2)), prevalence=float(y.mean()))
    return out


def operating_point(y: np.ndarray, p: np.ndarray, thr: float, side: str = "present") -> dict:
    """side='present': predict positive when p >= thr. side='absent': predict negative when p <= thr."""
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    if side == "present":
        pred = p >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        return {"thr": thr, "flagged": float(pred.mean()), "sens": tp / max(y.sum(), 1),
                "spec": float(((~pred) & (y == 0)).sum() / max((y == 0).sum(), 1)), "ppv": tp / max(tp + fp, 1)}
    pred_abs = p <= thr
    tn, fn = int((pred_abs & (y == 0)).sum()), int((pred_abs & (y == 1)).sum())
    return {"thr": thr, "flagged": float(pred_abs.mean()), "npv": tn / max(tn + fn, 1),
            "spec_absent": tn / max((y == 0).sum(), 1), "missed_pos_frac": fn / max(y.sum(), 1)}


MIN_TP = 10   # a precision target only counts if it is met while flagging at least this many true positives


def threshold_candidates(y: np.ndarray, p: np.ndarray, ppv_targets=(0.4, 0.5, 0.7, 0.8, 0.9),
                         miss_targets=(0.02, 0.05, 0.10)) -> list[dict]:
    """Present-band candidates by target PPV and sensitivity; absent-band candidates by target NPV
    and by the fraction of positives allowed below the threshold."""
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    rows = []
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    ppv = tp / k
    sens = tp / max(y.sum(), 1)
    for target in ppv_targets:
        ok = np.nonzero((ppv >= target) & (tp >= MIN_TP))[0]
        if len(ok):
            i = ok[-1]       # largest set that still reaches the target precision
            rows.append({"band": "present", "rule": f"PPV>={target}", **operating_point(y, p, float(ps[i]), "present")})
        else:
            rows.append({"band": "present", "rule": f"PPV>={target}", "thr": np.nan, "note": f"unreachable with >= {MIN_TP} true positives"})
    for target in (0.8, 0.9, 0.95):
        i = int(np.searchsorted(sens, target))
        if i < len(ps):
            rows.append({"band": "present", "rule": f"sens>={target}", **operating_point(y, p, float(ps[i]), "present")})
    asc = np.argsort(p)
    ya, pa = y[asc], p[asc]
    fn = np.cumsum(ya)
    kk = np.arange(1, len(ya) + 1)
    npv = 1 - fn / kk
    tn = np.cumsum(1 - ya)
    for target in (0.95, 0.98, 0.99):
        ok = np.nonzero((npv >= target) & (tn >= MIN_TP))[0]
        if len(ok):
            i = ok[-1]
            rows.append({"band": "absent", "rule": f"NPV>={target}", **operating_point(y, p, float(pa[i]), "absent")})
        else:
            rows.append({"band": "absent", "rule": f"NPV>={target}", "thr": np.nan, "note": "unreachable"})
    for miss in miss_targets:
        i = int(np.searchsorted(fn / max(y.sum(), 1), miss, side="right")) - 1
        if i >= 0:
            rows.append({"band": "absent", "rule": f"missed_pos<={miss}", **operating_point(y, p, float(pa[i]), "absent")})
    return rows
