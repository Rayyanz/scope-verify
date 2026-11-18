"""Pilot analysis used in the paper (Tables X-XII): bootstrap CIs (B=10,000, question clusters),
H1a Holm-corrected tests, exploratory comparisons, LSR, cost. Run from the repo root after scripts 01-04 on the val split.
Usage: python scripts/06_pilot_analysis.py  -> writes results.json"""
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
df = pd.read_csv("outputs/val_scores_nli-minicheck.csv")
cl = pd.read_json("outputs/val_claims_labeled.jsonl", lines=True)
key = ["qid","t","condition"]
W = df.pivot_table(index=key, columns=["verifier","scope"], values="score").reset_index()
W.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in W.columns]
W = W.merge(cl[key+["y_acc","y_pool","label","later_supported"]], on=key, how="left")
print("claims:", len(W), "ungrounded(acc):", int(W.y_acc.sum()), "ungrounded(pool):", int(W.y_pool.sum()))

def auroc(s, y):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(-s)   # high risk = low support
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)
def r95(s, y, ret=0.95):
    s = np.asarray(s); y = np.asarray(y)
    sup = np.sort(s[y == 0]); ung = s[y == 1]
    if len(sup) == 0 or len(ung) == 0: return np.nan
    th = sup[int(np.floor((1-ret)*len(sup)))]
    return float(np.mean(ung < th))

groups = W.groupby("qid").indices; keys = list(groups); idx = [groups[k] for k in keys]
rng = np.random.default_rng(13); B = 10000
samples = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(B)]
def boot(fn):
    point = fn(W)
    reps = np.array([fn(W.iloc[s]) for s in samples]); reps = reps[~np.isnan(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p = min(1.0, 2*min((reps <= 0).mean(), (reps >= 0).mean()))
    return point, lo, hi, p

V = ["nli","minicheck"]; S = ["loc","acc","traj","pool","pad"]
res = {"n_claims": len(W), "n_ung": int(W.y_acc.sum()), "n_questions": len(keys), "B": B}
tab = {}
for v in V:
    for s in S:
        a = boot(lambda d, c=f"{v}_{s}": auroc(d[c], d.y_acc))
        r = boot(lambda d, c=f"{v}_{s}": r95(d[c], d.y_acc))
        tab[f"{v}_{s}"] = {"auroc": a[:3], "r95": r[:3]}
res["table"] = tab
tests = {}
for v in V:
    for o in ["traj","pool"]:
        tests[f"H1a R95 {v} acc-{o}"] = boot(lambda d, v=v, o=o: r95(d[f"{v}_acc"], d.y_acc) - r95(d[f"{v}_{o}"], d.y_acc))
for v in V:
    tests[f"AUROC {v} acc-pool"] = boot(lambda d, v=v: auroc(d[f"{v}_acc"], d.y_acc) - auroc(d[f"{v}_pool"], d.y_acc))
    tests[f"R95 {v} loc-pad"] = boot(lambda d, v=v: r95(d[f"{v}_loc"], d.y_acc) - r95(d[f"{v}_pad"], d.y_acc))
for s in ["acc","pool"]:
    tests[f"R95 nli-minicheck @{s}"] = boot(lambda d, s=s: r95(d[f"nli_{s}"], d.y_acc) - r95(d[f"minicheck_{s}"], d.y_acc))
    tests[f"AUROC nli-minicheck @{s}"] = boot(lambda d, s=s: auroc(d[f"nli_{s}"], d.y_acc) - auroc(d[f"minicheck_{s}"], d.y_acc))
# H1c: gold definition acc vs pool, verifier input = acc and pool
for v in V:
    for s in ["acc","pool"]:
        tests[f"H1c AUROC {v}@{s} gold_acc-gold_pool"] = boot(lambda d, v=v, s=s: auroc(d[f"{v}_{s}"], d.y_acc) - auroc(d[f"{v}_{s}"], d.y_pool))
# Holm on the 4 primary H1a tests
prim = [k for k in tests if k.startswith("H1a")]
ps = np.array([tests[k][3] for k in prim]); order = np.argsort(ps); m = len(ps); run = 0; adj = {}
for r_, i in enumerate(order):
    run = max(run, (m - r_) * ps[i]); adj[prim[i]] = min(1.0, run)
res["tests"] = {k: {"diff": v[0], "ci": [v[1], v[2]], "p": v[3], "p_holm": adj.get(k)} for k, v in tests.items()}
# LSR
u = cl[cl.y_acc == 1].dropna(subset=["later_supported"])
Wl = u.assign(ls=u.later_supported.astype(int))
g = Wl.groupby("qid").indices; ks = list(g)
lsr = Wl.ls.mean(); reps = []
for _ in range(B):
    pick = np.concatenate([g[ks[i]] for i in rng.integers(0, len(ks), len(ks))]); reps.append(Wl.ls.values[pick].mean())
res["lsr"] = [lsr, *np.percentile(reps, [2.5, 97.5]), len(Wl)]
# per type recall (acc), descriptive
pt = {}
for v in V:
    for typ in ["MB","CC"]:
        sub = W[(W.label == typ) | (W.y_acc == 0)]
        pt[f"{v}_{typ}"] = [r95(sub[f"{v}_acc"], (sub.label == typ).astype(int)), int((sub.label == typ).sum())]
res["per_type"] = pt
res["by_condition"] = cl.groupby("condition").agg(n=("y_acc","size"), ung=("y_acc","sum")).reset_index().to_dict("records")
res["label_source"] = cl.label_source.value_counts().to_dict()
res["cost_sec_per_claim"] = df.groupby(["verifier","scope"]).seconds.mean().unstack().round(4).to_dict()
json.dump(res, open("results.json","w"), indent=1, default=float)
fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
for k, v in tab.items(): print(f"{k:16s} AUROC {fmt(v['auroc'])}   R95 {fmt(v['r95'])}")
for k, v in res["tests"].items(): print(f"{k:42s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p={v['p']:.3f}" + (f" holm={v['p_holm']:.3f}" if v['p_holm'] is not None else ""))
print("LSR", res["lsr"]); print("per type", pt); print(res["by_condition"]); print(res["cost_sec_per_claim"])
