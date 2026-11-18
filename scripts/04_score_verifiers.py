"""Score every (claim, scope, verifier) and meter cost (Algorithm 2).
All pairs of one verifier and scope are scored in a single batched call; the measured
batch cost is allocated to claims in proportion to their input length.
Usage: python scripts/04_score_verifiers.py --split test --verifiers nli minicheck
       python scripts/04_score_verifiers.py --split test --verifiers llm_judge   (separate process)"""
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # safe multi-GPU start

import argparse
import random

import pandas as pd
import yaml

from scopeverify.data import load_jsonl
from scopeverify.labels import format_evidence
from scopeverify.scopes import build_scopes
from scopeverify.verifiers import CostMeter


def make_verifier(name, cfg):
    from scopeverify import verifiers as V
    vc = cfg["verifiers"]
    return {"nli": lambda: V.NLIVerifier(vc["nli"]),
            "alignscore": lambda: V.AlignScoreVerifier(vc["alignscore_ckpt"]),
            "minicheck": lambda: V.MiniCheckVerifier(vc["minicheck"]),
            "llm_judge": lambda: V.LLMJudgeVerifier(vc["llm_judge"], vllm_kwargs=cfg.get("vllm"))}[name]()


def score_claims(verifier, vname, claims, scopes_of, scopes):
    """Return rows: one per (claim, scope) with score and allocated cost."""
    rows = []
    for scope in scopes:
        pairs, owner = [], []
        for ci, c in enumerate(claims):
            paras = scopes_of[ci][scope]
            if not paras:
                continue
            docs = [format_evidence(paras)] if vname == "llm_judge" else \
                   [p["title"] + ". " + p["text"] for p in paras]
            for d in docs:
                pairs.append((c["claim"], d))
                owner.append(ci)
        meter = CostMeter()
        scores = verifier.score_pairs([p[0] for p in pairs], [p[1] for p in pairs], meter) if pairs else []
        lens = [len(a) + len(b) for a, b in pairs]
        total_len = sum(lens) or 1
        ptoks = getattr(verifier, "last_prompt_tokens", None) if vname == "llm_judge" else None
        best = {}
        cost = {}
        for k, (ci, s) in enumerate(zip(owner, scores)):
            best[ci] = max(best.get(ci, 0.0), s)
            share = lens[k] / total_len
            c = cost.setdefault(ci, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0})
            c["seconds"] += meter.seconds * share
            if vname == "llm_judge":
                c["calls"] += 1
                c["tokens_in"] += ptoks[k] if ptoks else 0
                c["tokens_out"] += 1
            else:
                c["tokens_in"] += round(meter.tokens_in * share)
        for ci, c in enumerate(claims):
            cc = cost.get(ci, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0})
            rows.append({"qid": c["qid"], "t": c["t"], "condition": c["condition"], "label": c["label"],
                         "y_acc": c["y_acc"], "y_pool": c["y_pool"], "verifier": vname, "scope": scope,
                         "score": best.get(ci, 0.0), **cc})
        print(f"  {vname}/{scope}: {len(pairs)} pairs, {meter.seconds:.1f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--verifiers", nargs="+", default=["nli", "minicheck"])
    ap.add_argument("--judge_subsample", type=float, default=None,
                    help="fraction of claims scored by the LLM judge (use 1.0 on the val split)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    filler = load_jsonl("outputs/external_paragraphs.jsonl")[:2000]

    claims = load_jsonl(f"outputs/{args.split}_claims_labeled.jsonl")
    traces, metas = {}, {}
    for cond in cfg["conditions"]:
        for t in load_jsonl(f"outputs/{args.split}_{cond}_traces.jsonl"):
            traces[(cond, t["id"])] = t
        for m in load_jsonl(f"outputs/{args.split}_{cond}_meta.jsonl"):
            metas[(cond, m["id"])] = m

    rows = []
    for vname in args.verifiers:
        sub = claims
        if vname == "llm_judge":
            frac = args.judge_subsample if args.judge_subsample is not None else cfg["verifiers"]["llm_judge_subsample"]
            rng = random.Random(cfg["seed"])
            sub = [c for c in claims if rng.random() <= frac]
        scopes_of = [build_scopes(traces[(c["condition"], c["qid"])]["steps"], c["t"],
                                  metas[(c["condition"], c["qid"])]["pool"], filler) for c in sub]
        print(f"{vname}: {len(sub)} claims")
        v = make_verifier(vname, cfg)
        rows += score_claims(v, vname, sub, scopes_of, cfg["scopes"])
        del v
    out = f"outputs/{args.split}_scores_{'-'.join(args.verifiers)}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} score rows written to {out}")


if __name__ == "__main__":   # required: vLLM starts worker processes with "spawn"
    main()
