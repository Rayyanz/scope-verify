"""Budget-constrained verification cascade (paper Eq. 10-12, Algorithm 3)."""
import numpy as np

from .metrics import auroc


def cascade_scores(cheap, judge, lo, hi):
    cheap, judge = np.asarray(cheap, float), np.asarray(judge, float)
    esc = (cheap >= lo) & (cheap <= hi)
    return np.where(esc, judge, cheap), esc


def mean_cost(esc, k_cheap, k_judge):
    return float(k_cheap + k_judge * np.mean(esc))


def calibrate_band(cheap, judge, y, k_cheap, k_judge, delta, step=0.05):
    """Grid search for (lo, hi) minimising expected cost subject to
    AUROC(cascade) >= AUROC(judge) - delta. Returns dict or None if infeasible."""
    target = auroc(judge, y) - delta
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    best = None
    for lo in grid:
        for hi in grid[grid >= lo]:
            s, esc = cascade_scores(cheap, judge, lo, hi)
            a = auroc(s, y)
            c = mean_cost(esc, k_cheap, k_judge)
            if a >= target and (best is None or c < best["cost"]):
                best = {"lo": float(lo), "hi": float(hi), "auroc": a, "cost": c,
                        "escalation_rate": float(np.mean(esc))}
    return best
