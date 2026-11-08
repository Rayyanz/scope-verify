"""Segment steps, decontextualise, construct labels (HLEP) and judge the rest.
Also computes y_pool (gold scope = pool) and the later-supported flag for LSR.
Usage: python scripts/03_build_claims_and_labels.py --split test"""
import argparse

import yaml

from scopeverify.claims import build_claims
from scopeverify.data import load_jsonl, save_jsonl
from scopeverify.labels import (JUDGE_PROMPT, binary, construct_label, format_evidence,
                                parse_judge)
from scopeverify.llm import VLLMText
from scopeverify.scopes import build_scopes, retrieved_before

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="test")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
judge = VLLMText(cfg["label_judge"], max_tokens=256, vllm_kwargs=cfg.get("vllm"))


def judge_label(claim, paragraphs):
    out = judge([JUDGE_PROMPT.format(evidence=format_evidence(paragraphs), claim=claim)], max_tokens=4)[0]
    return parse_judge(out)


rows = []
for cond in cfg["conditions"]:
    metas = {m["id"]: m for m in load_jsonl(f"outputs/{args.split}_{cond}_meta.jsonl")}
    for tr in load_jsonl(f"outputs/{args.split}_{cond}_traces.jsonl"):
        meta = metas[tr["id"]]
        for c in build_claims(tr, llm_fn=judge):
            sc = build_scopes(tr["steps"], c["t"], meta["pool"], filler=[])
            lab = construct_label(c["claim"], meta, sc["acc"], retrieved_before(tr["steps"], c["t"]))
            source = "construction" if lab else "judge"
            lab = lab or judge_label(c["claim"], sc["acc"])
            y_acc = binary(lab)
            y_pool = binary(judge_label(c["claim"], sc["pool"]))
            later = binary(judge_label(c["claim"], sc["traj"])) == 0 if y_acc == 1 else None
            rows.append({**c, "condition": cond, "label": lab, "label_source": source,
                         "y_acc": y_acc, "y_pool": y_pool, "later_supported": later})
save_jsonl(rows, f"outputs/{args.split}_claims_labeled.jsonl")
print(f"{len(rows)} labelled claims")
