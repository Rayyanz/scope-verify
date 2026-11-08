"""Question-level (cluster) bootstrap, Holm correction, non-inferiority."""
import numpy as np


def cluster_bootstrap(df, stat_fn, cluster_col="qid", n_boot=10_000, seed=13):
    """Resample clusters with replacement; return (point, lo, hi, replicates).
    stat_fn(df) -> float. Rows of each cluster are kept together."""
    rng = np.random.default_rng(seed)
    groups = {k: idx for k, idx in df.groupby(cluster_col).indices.items()}
    keys = np.array(list(groups))
    point = stat_fn(df)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([groups[k] for k in pick])
        reps[b] = stat_fn(df.iloc[rows])
    reps = reps[~np.isnan(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(point), float(lo), float(hi), reps


def bootstrap_p_two_sided(reps):
    """Approximate two-sided p-value for H0: statistic = 0."""
    reps = np.asarray(reps)
    return float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def non_inferior(lo, delta):
    """H3: reject H0 (inferiority) if the lower CI bound of AUROC(P3)-AUROC(P1) > -delta."""
    return lo > -delta
