"""Compute RQ1-RQ3 results with question-level bootstrap CIs and the Pareto plot.
Usage: python scripts/05_analyze.py --split test --val_split val --n_boot 10000"""
import argparse
import itertools
import json

import matplotlib
import numpy as np
import pandas as pd

from scopeverify.cascade import calibrate_band, cascade_scores, mean_cost
from scopeverify.metrics import auroc, later_supported_rate, recall_at_retention
from scopeverify.stats import bootstrap_p_two_sided, cluster_bootstrap, holm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--val_split", default="val")
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--delta", type=float, required=True, help="cascade tolerance fixed after pilot")
ap.add_argument("--cheap", default="minicheck")
args = ap.parse_args()

df = pd.read_csv(f"outputs/{args.split}_scores.csv").dropna(subset=["y_acc"])
df["uid"] = df["condition"] + "|" + df["qid"]           # cluster = question x condition
df["cluster"] = df["qid"]                                # resample questions (all conditions)
res = {}


def ci(sub, fn):
    p, lo, hi, reps = cluster_bootstrap(sub, fn, "cluster", args.n_boot)
    return {"point": p, "lo": lo, "hi": hi}, reps


# ---- RQ1: AUROC and R@95 by verifier x scope (gold = acc) ----
table, reps_store = [], {}
for (v, sc), sub in df.groupby(["verifier", "scope"]):
    a, ra = ci(sub, lambda d: auroc(d["score"], d["y_acc"]))
    r, rr = ci(sub, lambda d: recall_at_retention(d["score"], d["y_acc"]))
    table.append({"verifier": v, "scope": sc, "auroc": a, "r95": r, "n": len(sub)})
res["rq1_table"] = table

# H1a: R@95(acc) - R@95(traj / pool), paired by step
tests = []
for v, sub in df.groupby("verifier"):
    wide = sub.pivot_table(index=["cluster", "uid", "t"], columns="scope", values="score").reset_index()
    y = sub.drop_duplicates(["uid", "t"]).set_index(["uid", "t"])["y_acc"]
    wide["y"] = [y.loc[(u, t)] for u, t in zip(wide["uid"], wide["t"])]
    for other in ["traj", "pool"]:
        if other not in wide:
            continue
        fn = lambda d, o=other: recall_at_retention(d["acc"], d["y"]) - recall_at_retention(d[o], d["y"])
        p, lo, hi, reps = cluster_bootstrap(wide, fn, "cluster", args.n_boot)
        tests.append({"test": f"H1a {v} acc-{other}", "diff": p, "lo": lo, "hi": hi,
                      "p": bootstrap_p_two_sided(reps)})
# H1b: rank reversal between pairs of verifiers across scopes
for sc_a, sc_b in itertools.combinations(sorted(df["scope"].unique()), 2):
    for v1, v2 in itertools.combinations(sorted(df["verifier"].unique()), 2):
        out = {}
        for sc in (sc_a, sc_b):
            s1 = df[(df.verifier == v1) & (df.scope == sc)].set_index(["uid", "t"])
            s2 = df[(df.verifier == v2) & (df.scope == sc)].set_index(["uid", "t"])
            j = s1[["score", "y_acc", "cluster"]].join(s2[["score"]], rsuffix="_2", how="inner").reset_index()
            if len(j) < 20:
                break
            fn = lambda d: auroc(d["score"], d["y_acc"]) - auroc(d["score_2"], d["y_acc"])
            out[sc] = cluster_bootstrap(j, fn, "cluster", args.n_boot)[:3]
        if len(out) == 2:
            (pa, la, ha), (pb, lb, hb) = out[sc_a], out[sc_b]
            reversal = (la > 0 and hb < 0) or (ha < 0 and lb > 0)
            tests.append({"test": f"H1b {v1} vs {v2}: {sc_a}/{sc_b}", "diff_a": pa, "ci_a": [la, ha],
                          "diff_b": pb, "ci_b": [lb, hb], "reversal": bool(reversal)})
res["tests"] = tests
ps = [t["p"] for t in tests if "p" in t]
for t, padj in zip([t for t in tests if "p" in t], holm(ps) if ps else []):
    t["p_holm"] = padj

# ---- RQ2: R@95 per perturbation-induced type (scope = acc) ----
rq2 = []
acc = df[df.scope == "acc"]
for v, sub in acc.groupby("verifier"):
    for typ in ["MB", "IE", "CC"]:
        s = sub[(sub.label == typ) | (sub.y_acc == 0)]
        if (s.label == typ).sum() < 5:
            continue
        r, _ = ci(s, lambda d: recall_at_retention(d["score"], d["y_acc"]))
        rq2.append({"verifier": v, "type": typ, "r95": r, "n_type": int((s.label == typ).sum())})
res["rq2"] = rq2

# ---- LSR (needs later_supported from 03) ----
try:
    cl = pd.read_json(f"outputs/{args.split}_claims_labeled.jsonl", lines=True)
    cl = cl[cl.y_acc == 1].dropna(subset=["later_supported"])
    cl["cluster"] = cl["qid"]
    res["lsr"] = ci(cl.assign(y=1), lambda d: later_supported_rate(d["y"], d["later_supported"].astype(int)))[0]
except (FileNotFoundError, ValueError):
    pass

# ---- RQ3: cascade calibrated on validation, evaluated on test (scope = acc) ----
def paired(frame, cheap):
    a = frame[(frame.scope == "acc") & (frame.verifier == cheap)].set_index(["uid", "t"])
    b = frame[(frame.scope == "acc") & (frame.verifier == "llm_judge")].set_index(["uid", "t"])
    return a[["score", "y_acc", "cluster", "seconds", "tokens_in"]].join(
        b[["score", "seconds", "tokens_in"]], rsuffix="_j", how="inner").reset_index()

try:
    val = pd.read_csv(f"outputs/{args.val_split}_scores.csv").dropna(subset=["y_acc"])
    val["uid"] = val["condition"] + "|" + val["qid"]; val["cluster"] = val["qid"]
    pv, pt = paired(val, args.cheap), paired(df, args.cheap)
    kc, kj = pv["seconds"].mean(), pv["seconds_j"].mean()
    band = calibrate_band(pv["score"], pv["score_j"], pv["y_acc"], kc, kj, args.delta)
    if band:
        def diff(d):
            s, _ = cascade_scores(d["score"], d["score_j"], band["lo"], band["hi"])
            return auroc(s, d["y_acc"]) - auroc(d["score_j"], d["y_acc"])
        p, lo, hi, _ = cluster_bootstrap(pt, diff, "cluster", args.n_boot)
        s, esc = cascade_scores(pt["score"], pt["score_j"], band["lo"], band["hi"])
        res["rq3"] = {"band": band, "test_diff": [p, lo, hi], "non_inferior": lo > -args.delta,
                      "escalation_rate": float(esc.mean()),
                      "tokens_judge_P1": float(pt["tokens_in_j"].mean()),
                      "tokens_judge_P3": float((pt["tokens_in_j"] * esc).mean())}
        pts = {"P1 judge": (pt["tokens_in_j"].mean(), auroc(pt["score_j"], pt["y_acc"])),
               f"P2 {args.cheap}": (pt["tokens_in"].mean(), auroc(pt["score"], pt["y_acc"])),
               "P3 cascade": (pt["tokens_in"].mean() + (pt["tokens_in_j"] * esc).mean(),
                              auroc(s, pt["y_acc"]))}
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for name, (x, yv) in pts.items():
            ax.scatter(x, yv); ax.annotate(name, (x, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("verification tokens per step"); ax.set_ylabel("AUROC"); ax.set_xscale("log")
        fig.tight_layout(); fig.savefig("figs/pareto.pdf")
except FileNotFoundError:
    print("no validation scores: run the pipeline on the val split to calibrate the cascade")

json.dump(res, open(f"outputs/{args.split}_results.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float)[:3000])
