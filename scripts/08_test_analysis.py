"""Test-set analysis used in the paper (Tables X-XIV): bootstrap CIs (B=5,000, question clusters), H1a Holm-corrected
tests, AUROC scope differences, length control, H1b, H1c, RQ2 per-type recall, LSR, generator statistics.
Run from the repo root after scripts 01-04 (all three verifiers) on the test split; writes results_test.json."""
import json, glob, numpy as np, pandas as pd, time
from scipy.stats import rankdata
B = 5000
def load(prefix):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{prefix}_scores_*.csv"))])
    cl = pd.read_json(f"{prefix}_claims_labeled.jsonl", lines=True)
    key = ["qid","t","condition"]
    W = df.pivot_table(index=key, columns=["verifier","scope"], values="score").reset_index()
    W.columns = ["_".join(c) if isinstance(c,tuple) and c[1] else (c[0] if isinstance(c,tuple) else c) for c in W.columns]
    C = df.pivot_table(index=key, columns=["verifier","scope"], values="seconds").reset_index()
    C.columns = [("sec_"+"_".join(c)) if isinstance(c,tuple) and c[1] else (c[0] if isinstance(c,tuple) else c) for c in C.columns]
    W = W.merge(C, on=key).merge(cl[key+["y_acc","y_pool","label","label_source","later_supported"]], on=key)
    return W, df, cl
def auroc(s,y):
    y=np.asarray(y); s=np.asarray(s); m=~np.isnan(s); y=y[m]; s=s[m]; n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=rankdata(-s); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def thr95(s,y):
    sup=np.sort(np.asarray(s)[np.asarray(y)==0]); return sup[int(np.floor(0.05*len(sup)))]
def r95(s,y):
    s=np.asarray(s); y=np.asarray(y)
    if y.sum()==0 or (y==0).sum()==0: return np.nan
    return float(np.mean(s[y==1] < thr95(s,y)))
W, df, cl = load("outputs/test")
print("claims",len(W),"ungrounded",int(W.y_acc.sum()),"questions",W.qid.nunique())
g=W.groupby("qid").indices; ks=list(g); idx=[g[k] for k in ks]
rng=np.random.default_rng(13); RIDX=[rng.integers(0,len(idx),len(idx)) for _ in range(B)]; samples=[np.concatenate([idx[i] for i in r]) for r in RIDX]
class View(dict):
    def __getattr__(self,k): return self[k]
def mk(d): return {c:d[c].values for c in d.columns}
ARR={}
def boot(fn, d=W):
    key=id(d)
    if key not in ARR: ARR[key]=(mk(d), d.reset_index(drop=True).groupby("qid").indices)
    A,gi=ARR[key]; kk=list(gi); ii=[gi[k] for k in kk]
    smp=samples if d is W else [np.concatenate([ii[i] for i in r]) for r in RIDX]
    pt=fn(View(A)); reps=np.array([fn(View({c:v[s_] for c,v in A.items()})) for s_ in smp]); reps=reps[~np.isnan(reps)]
    lo,hi=np.percentile(reps,[2.5,97.5]); p=min(1.0,2*min((reps<=0).mean(),(reps>=0).mean())); return [pt,lo,hi,p]
t0=time.time()
V=["nli","minicheck","llm_judge"]; S=["loc","acc","traj","pool","pad"]; R={"table":{},"tests":{}}
for v in V:
    for s in S:
        c=f"{v}_{s}"
        R["table"][c]={"auroc":boot(lambda d,c=c:auroc(d[c],d.y_acc))[:3],"r95":boot(lambda d,c=c:r95(d[c],d.y_acc))[:3],"sec":W[f"sec_{c}"].mean()}
    print(v, round(time.time()-t0))
T=R["tests"]
for v in V:
    for o in ["traj","pool"]:
        T[f"H1a {v} acc-{o}"]=boot(lambda d,v=v,o=o: r95(d[f"{v}_acc"],d.y_acc)-r95(d[f"{v}_{o}"],d.y_acc))
prim=[k for k in T if k.startswith("H1a")]; ps=np.array([T[k][3] for k in prim]); o=np.argsort(ps); run=0
for r_,i in enumerate(o): run=max(run,(len(ps)-r_)*ps[i]); T[prim[i]].append(min(1,run))
for v in V:
    T[f"AUROC {v} acc-pool"]=boot(lambda d,v=v: auroc(d[f"{v}_acc"],d.y_acc)-auroc(d[f"{v}_pool"],d.y_acc))
    T[f"AUROC {v} acc-traj"]=boot(lambda d,v=v: auroc(d[f"{v}_acc"],d.y_acc)-auroc(d[f"{v}_traj"],d.y_acc))
    T[f"R95 {v} loc-pad"]=boot(lambda d,v=v: r95(d[f"{v}_loc"],d.y_acc)-r95(d[f"{v}_pad"],d.y_acc))
for s in ["acc","pool"]:
    T[f"H1b R95 nli-minicheck @{s}"]=boot(lambda d,s=s: r95(d[f"nli_{s}"],d.y_acc)-r95(d[f"minicheck_{s}"],d.y_acc))
    T[f"H1b AUROC nli-minicheck @{s}"]=boot(lambda d,s=s: auroc(d[f"nli_{s}"],d.y_acc)-auroc(d[f"minicheck_{s}"],d.y_acc))
    for o2 in ["nli","minicheck"]:
        T[f"AUROC judge-{o2} @{s}"]=boot(lambda d,s=s,o2=o2: auroc(d[f"llm_judge_{s}"],d.y_acc)-auroc(d[f"{o2}_{s}"],d.y_acc))
dp=W.dropna(subset=["y_pool"]).reset_index(drop=True)
for v in ["nli","minicheck"]:
    for s in ["acc","pool"]:
        T[f"H1c AUROC {v}@{s} gacc-gpool"]=boot(lambda d,v=v,s=s: auroc(d[f"{v}_{s}"],d.y_acc)-auroc(d[f"{v}_{s}"],d.y_pool), dp)
print("tests", round(time.time()-t0))
# RQ2 per type: recall at the R95 threshold (threshold from all supported claims), MB vs CC and UNG(judge)
def rec_type(d,c,typ):
    th=thr95(d[c],d.y_acc); x=np.asarray(d[c])[np.asarray(d["label"])==typ]
    return float(np.mean(x<th)) if len(x) else np.nan
R["rq2"]={}
for v in ["nli","minicheck"]:
    for s in ["acc","pool"]:
        for typ in ["MB","CC","UNG"]:
            R["rq2"][f"{v}_{s}_{typ}"]=boot(lambda d,v=v,s=s,typ=typ: rec_type(d,f"{v}_{s}",typ))[:3]
    R["rq2"][f"{v}_acc_MB-CC"]=boot(lambda d,v=v: rec_type(d,f"{v}_acc","MB")-rec_type(d,f"{v}_acc","CC"))
    R["rq2"][f"{v}_MB acc-pool"]=boot(lambda d,v=v: rec_type(d,f"{v}_acc","MB")-rec_type(d,f"{v}_pool","MB"))
    R["rq2"][f"{v}_CC acc-pool"]=boot(lambda d,v=v: rec_type(d,f"{v}_acc","CC")-rec_type(d,f"{v}_pool","CC"))
print("rq2", round(time.time()-t0))
u=W[W.y_acc==1].dropna(subset=["later_supported"]).reset_index(drop=True); u["ls"]=u.later_supported.astype(int)
gu=u.groupby("qid").indices; ku=list(gu); reps=[u.ls.values[np.concatenate([gu[ku[i]] for i in rng.integers(0,len(ku),len(ku))])].mean() for _ in range(B)]
R["lsr"]=[u.ls.mean(),*np.percentile(reps,[2.5,97.5]),len(u)]
R["labels"]=pd.crosstab(cl.condition,cl.label).to_dict(); R["label_source"]=cl.label_source.value_counts().to_dict()
R["gold_disagree"]={"n":int(len(dp)),"agree":float((dp.y_acc==dp.y_pool).mean()),"acc1_pool0":int(((dp.y_acc==1)&(dp.y_pool==0)).sum()),"acc0_pool1":int(((dp.y_acc==0)&(dp.y_pool==1)).sum())}
R["pairs"]={f"{a}_{b}":int(n) for (a,b),n in df.groupby(["verifier","scope"]).size().items()}
# generator stats
def norm(x):
    import re,string
    x=(x or "").lower(); x="".join(ch for ch in x if ch not in set(string.punctuation)); x=re.sub(r"\b(a|an|the)\b"," ",x); return " ".join(x.split())
gen={}
for cond in ["clean","removed","counterfactual","closedbook"]:
    tr=[json.loads(l) for l in open(f"outputs/test_{cond}_traces.jsonl")]
    golds=[json.loads(l) for l in open("outputs/test.jsonl")]; al={g["id"]:[g["answer"]]+g.get("answer_aliases",[]) for g in golds}
    em=np.mean([any(norm(t["pred_answer"])==norm(a) for a in al[t["id"]]) for t in tr])*100
    gen[cond]={"n":len(tr),"em":em,"steps":np.mean([len(t["steps"]) for t in tr]),"noans":sum(1 for t in tr if not t["pred_answer"])}
R["gen"]=gen
json.dump(R,open("results_test.json","w"),indent=1,default=float)
print("done", round(time.time()-t0))
