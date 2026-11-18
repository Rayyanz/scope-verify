"""Pilot analysis of the LLM judge and the two-fold calibrated cascade (paper Table XIII).
Run from the repo root after scripts 01-04 (all three verifiers) on the val split."""
import json, numpy as np, pandas as pd, glob
from scipy.stats import rankdata
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("outputs/val_scores_*.csv"))])
cl = pd.read_json("outputs/val_claims_labeled.jsonl", lines=True)
key = ["qid","t","condition"]
W = df.pivot_table(index=key, columns=["verifier","scope"], values="score").reset_index()
W.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in W.columns]
C = df.pivot_table(index=key, columns=["verifier","scope"], values="seconds").reset_index()
C.columns = [("sec_"+"_".join(c)) if (isinstance(c, tuple) and c[1]) else (c[0] if isinstance(c, tuple) else c) for c in C.columns]
W = W.merge(C, on=key).merge(cl[key+["y_acc","y_pool","label"]], on=key)
def auroc(s, y):
    y=np.asarray(y); s=np.asarray(s); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=rankdata(-s); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def r95(s,y,ret=.95):
    s=np.asarray(s); y=np.asarray(y); sup=np.sort(s[y==0]); ung=s[y==1]
    if len(sup)==0 or len(ung)==0: return np.nan
    return float(np.mean(ung < sup[int(np.floor((1-ret)*len(sup)))]))
groups=W.groupby("qid").indices; keys=list(groups); idx=[groups[k] for k in keys]
rng=np.random.default_rng(13); B=10000
samples=[np.concatenate([idx[i] for i in rng.integers(0,len(idx),len(idx))]) for _ in range(B)]
def boot(fn, data=None):
    d = W if data is None else data
    pt=fn(d); reps=np.array([fn(d.iloc[s]) for s in samples]); reps=reps[~np.isnan(reps)]
    lo,hi=np.percentile(reps,[2.5,97.5]); p=min(1.0,2*min((reps<=0).mean(),(reps>=0).mean())); return pt,lo,hi,p
S=["loc","acc","traj","pool","pad"]; out={}
for s in S:
    out[f"llm_judge_{s}"]={"auroc":boot(lambda d,c=f"llm_judge_{s}":auroc(d[c],d.y_acc))[:3],"r95":boot(lambda d,c=f"llm_judge_{s}":r95(d[c],d.y_acc))[:3]}
    print(s, "AUROC %.3f [%.3f,%.3f]"%out[f"llm_judge_{s}"]["auroc"], "R95 %.3f [%.3f,%.3f]"%out[f"llm_judge_{s}"]["r95"], "sec/claim %.3f"%W[f"sec_llm_judge_{s}"].mean())
tests={}
for v in ["nli","minicheck","llm_judge"]:
    for o in ["traj","pool"]:
        tests[f"H1a {v} acc-{o}"]=boot(lambda d,v=v,o=o: r95(d[f"{v}_acc"],d.y_acc)-r95(d[f"{v}_{o}"],d.y_acc))
prim=list(tests); ps=np.array([tests[k][3] for k in prim]); order=np.argsort(ps); m=len(ps); run=0; holm={}
for r_,i in enumerate(order): run=max(run,(m-r_)*ps[i]); holm[prim[i]]=min(1,run)
tests["judge AUROC acc-pool"]=boot(lambda d: auroc(d.llm_judge_acc,d.y_acc)-auroc(d.llm_judge_pool,d.y_acc))
for s in ["acc","pool"]:
    for o in ["nli","minicheck"]:
        tests[f"AUROC judge-{o} @{s}"]=boot(lambda d,s=s,o=o: auroc(d[f"llm_judge_{s}"],d.y_acc)-auroc(d[f"{o}_{s}"],d.y_acc))
        tests[f"R95 judge-{o} @{s}"]=boot(lambda d,s=s,o=o: r95(d[f"llm_judge_{s}"],d.y_acc)-r95(d[f"{o}_{s}"],d.y_acc))
for k,v in tests.items(): print(f"{k:28s} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] p={v[3]:.3f}"+(f" holm={holm[k]:.3f}" if k in holm else ""))
# ---- cascade, 2-fold by question (band calibrated on one half, evaluated on the other) ----
delta=0.02; grid=np.round(np.arange(0,1.0001,0.05),2)
qs=np.array(keys); rng2=np.random.default_rng(7); rng2.shuffle(qs); folds=[set(qs[:len(qs)//2]), set(qs[len(qs)//2:])]
casc={}
for cheap in ["minicheck","nli"]:
    W[f"casc_{cheap}"]=np.nan; W[f"esc_{cheap}"]=np.nan; bands=[]
    for f in range(2):
        cal=W[W.qid.isin(folds[f])]; ev=W.qid.isin(folds[1-f])
        target=auroc(cal.llm_judge_acc,cal.y_acc)-delta; best=None
        for lo in grid:
            for hi in grid[grid>=lo]:
                e=(cal[f"{cheap}_acc"]>=lo)&(cal[f"{cheap}_acc"]<=hi); s=np.where(e,cal.llm_judge_acc,cal[f"{cheap}_acc"])
                a=auroc(s,cal.y_acc); c=cal[f"sec_{cheap}_acc"].mean()+cal.sec_llm_judge_acc.mean()*e.mean()
                if a>=target and (best is None or c<best[2]): best=(lo,hi,c,e.mean())
        bands.append(best)
        e=(W.loc[ev,f"{cheap}_acc"]>=best[0])&(W.loc[ev,f"{cheap}_acc"]<=best[1])
        W.loc[ev,f"casc_{cheap}"]=np.where(e,W.loc[ev,"llm_judge_acc"],W.loc[ev,f"{cheap}_acc"]); W.loc[ev,f"esc_{cheap}"]=e.astype(float)
    d=boot(lambda d,c=cheap: auroc(d[f"casc_{c}"],d.y_acc)-auroc(d.llm_judge_acc,d.y_acc))
    esc=W[f"esc_{cheap}"].mean()
    cost_c=W[f"sec_{cheap}_acc"].mean()+W.sec_llm_judge_acc.mean()*esc; cost_j=W.sec_llm_judge_acc.mean()
    casc[cheap]={"bands":bands,"auroc":auroc(W[f"casc_{cheap}"],W.y_acc),"diff":d,"esc":esc,"cost":cost_c,"cost_judge":cost_j,"calls_frac":esc}
    print(f"\nCASCADE {cheap}->judge: bands {[(b[0],b[1]) for b in bands]} AUROC {casc[cheap]['auroc']:.3f} vs judge {auroc(W.llm_judge_acc,W.y_acc):.3f} | diff {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] | non-inferior(delta=.02): {d[1]>-delta} | escalation {esc:.1%} | sec/claim {cost_c:.3f} vs {cost_j:.3f} ({cost_c/cost_j:.0%})")
json.dump({"judge":out,"tests":{k:list(v) for k,v in tests.items()},"holm":holm,"cascade":casc},open("results2.json","w"),indent=1,default=float)
