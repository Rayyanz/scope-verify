"""Detection metrics (paper Table VI). Convention: score = support (higher =
supported); y = 1 means ungrounded (positive class)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _arr(s, y):
    return np.asarray(s, float), np.asarray(y, int)


def auroc(scores, y):
    s, y = _arr(scores, y)
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, -s))


def auprc(scores, y):
    s, y = _arr(scores, y)
    return float(average_precision_score(y, -s)) if y.sum() else float("nan")


def recall_at_retention(scores, y, retention=0.95):
    """theta = largest threshold keeping >= `retention` of supported steps at
    s >= theta; return fraction of ungrounded steps with s < theta."""
    s, y = _arr(scores, y)
    sup, ung = np.sort(s[y == 0]), s[y == 1]
    if len(sup) == 0 or len(ung) == 0:
        return float("nan")
    n_flag_allowed = int(np.floor((1 - retention) * len(sup)))
    theta = sup[n_flag_allowed]  # supported scores below theta: at most n_flag_allowed
    return float(np.mean(ung < theta))


def confusion(scores, y, theta):
    s, y = _arr(scores, y)
    pred = (s < theta).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def f1_at(scores, y, theta):
    tp, fp, fn, _ = confusion(scores, y, theta)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def balanced_accuracy(scores, y, theta):
    tp, fp, fn, tn = confusion(scores, y, theta)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return 0.5 * (tpr + tnr)


def tune_threshold(scores, y, grid=None):
    """Threshold maximising step F1 (fit on validation only)."""
    grid = np.linspace(0, 1, 101) if grid is None else grid
    return float(max(grid, key=lambda t: f1_at(scores, y, t)))


def later_supported_rate(y_acc, entailed_by_traj):
    """Eq. 9: share of acc-ungrounded steps entailed by E^traj."""
    y, e = np.asarray(y_acc, int), np.asarray(entailed_by_traj, int)
    return float(e[y == 1].mean()) if (y == 1).any() else float("nan")


def question_level_f1(df, score_col, theta, qcol="qid", ycol="y"):
    g = df.assign(flag=(df[score_col] < theta).astype(int)).groupby(qcol)
    pred, gold = g["flag"].max().values, g[ycol].max().values
    tp = int(((pred == 1) & (gold == 1)).sum()); fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0
