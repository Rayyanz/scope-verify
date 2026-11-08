"""Score every (claim, scope, verifier) and meter cost (Algorithm 2).
Usage: python scripts/04_score_verifiers.py --split test --verifiers nli minicheck alignscore"""
import argparse
import random

import pandas as pd
import yaml

from scopeverify.data import load_jsonl
from scopeverify.scopes import build_scopes
from scopeverify.verifiers import (AlignScoreVerifier, CostMeter, LLMJudgeVerifier,
                                   MiniCheckVerifier, NLIVerifier, score_scope)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--verifiers", nargs="+", default=["nli", "minicheck", "alignscore"])
ap.add_argument("--judge_subsample", type=float, default=None, help="use 1.0 on the val split (cascade calibration)")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
vc = cfg["verifiers"]
filler = load_jsonl("outputs/external_paragraphs.jsonl")[:2000]

make = {"nli": lambda: NLIVerifier(vc["nli"]),
        "alignscore": lambda: AlignScoreVerifier(vc["alignscore_ckpt"]),
        "minicheck": lambda: MiniCheckVerifier(vc["minicheck"]),
        "llm_judge": lambda: LLMJudgeVerifier(vc["llm_judge"], vllm_kwargs=cfg.get("vllm"))}

claims = load_jsonl(f"outputs/{args.split}_claims_labeled.jsonl")
traces, metas = {}, {}
for cond in cfg["conditions"]:
    for t in load_jsonl(f"outputs/{args.split}_{cond}_traces.jsonl"):
        traces[(cond, t["id"])] = t
    for m in load_jsonl(f"outputs/{args.split}_{cond}_meta.jsonl"):
        metas[(cond, m["id"])] = m

rng = random.Random(cfg["seed"])
rows = []
for vname in args.verifiers:
    v = make[vname]()
    chunk = "all" if vname == "llm_judge" else "paragraph"
    for c in claims:
        frac = args.judge_subsample if args.judge_subsample is not None else vc["llm_judge_subsample"]
        if vname == "llm_judge" and rng.random() > frac:
            continue
        key = (c["condition"], c["qid"])
        sc = build_scopes(traces[key]["steps"], c["t"], metas[key]["pool"], filler)
        for scope in cfg["scopes"]:
            m = CostMeter()
            s = score_scope(v, c["claim"], sc[scope], chunk=chunk, meter=m)
            rows.append({"qid": c["qid"], "t": c["t"], "condition": c["condition"],
                         "label": c["label"], "y_acc": c["y_acc"], "y_pool": c["y_pool"],
                         "verifier": vname, "scope": scope, "score": s, "calls": m.calls,
                         "tokens_in": m.tokens_in, "tokens_out": m.tokens_out, "seconds": m.seconds})
    del v
pd.DataFrame(rows).to_csv(f"outputs/{args.split}_scores.csv", index=False)
print(f"{len(rows)} score rows written")
