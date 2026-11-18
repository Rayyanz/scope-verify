"""Segment steps, decontextualise, construct labels (HLEP) and judge the rest.
Also computes y_pool (gold scope = pool) and the later-supported flag for LSR.
All LLM calls are batched (one vLLM call per stage).
Usage: python scripts/03_build_claims_and_labels.py --split val"""
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # safe multi-GPU start

import argparse

import yaml

from scopeverify.claims import DECONTEXT_PROMPT, FILTER_PROMPT, step_text
from scopeverify.data import load_jsonl, save_jsonl
from scopeverify.labels import (JUDGE_PROMPT, binary, construct_label, format_evidence,
                                parse_judge)
from scopeverify.scopes import build_scopes, retrieved_before


def build_rows(cfg, split, llm):
    """llm(prompts, max_tokens) -> list[str]. Returns labelled claim rows."""
    items = []
    for cond in cfg["conditions"]:
        metas = {m["id"]: m for m in load_jsonl(f"outputs/{split}_{cond}_meta.jsonl")}
        for tr in load_jsonl(f"outputs/{split}_{cond}_traces.jsonl"):
            texts = [step_text(s) for s in tr["steps"]]
            for i, (s, txt) in enumerate(zip(tr["steps"], texts)):
                if txt:
                    items.append({"cond": cond, "tr": tr, "meta": metas[tr["id"]], "t": s["t"],
                                  "raw": txt, "hist": "\n".join(t for t in texts[:i] if t) or "(none)"})
    print(f"{len(items)} non-empty steps")

    # 1) filter non-claims (batched)
    outs = llm([FILTER_PROMPT.format(text=it["raw"]) for it in items], 8)
    items = [it for it, o in zip(items, outs) if "NOT" not in o.upper()]
    print(f"{len(items)} steps kept as claims")

    # 2) decontextualise (batched)
    outs = llm([DECONTEXT_PROMPT.format(question=it["tr"]["question"], history=it["hist"],
                                        text=it["raw"]) for it in items], 256)
    for it, o in zip(items, outs):
        it["claim"] = o.strip() or it["raw"]

    # 3) scopes + labels by construction
    for it in items:
        it["sc"] = build_scopes(it["tr"]["steps"], it["t"], it["meta"]["pool"], filler=[])
        ids = retrieved_before(it["tr"]["steps"], it["t"])
        it["label"] = construct_label(it["claim"], it["meta"], it["sc"]["acc"], ids)
        it["source"] = "construction" if it["label"] else "judge"

    # 4) judge: acc (where needed), pool and traj (all), batched
    def judge(scope, subset):
        outs = llm([JUDGE_PROMPT.format(evidence=format_evidence(it["sc"][scope]), claim=it["claim"])
                    for it in subset], 4)
        return [parse_judge(o) for o in outs]

    need = [it for it in items if it["label"] is None]
    for it, lab in zip(need, judge("acc", need)):
        it["label"] = lab
    for it, lab in zip(items, judge("pool", items)):
        it["label_pool"] = lab
    for it, lab in zip(items, judge("traj", items)):
        it["label_traj"] = lab

    rows = []
    for it in items:
        y_acc = binary(it["label"])
        rows.append({"qid": it["tr"]["id"], "t": it["t"], "raw": it["raw"], "claim": it["claim"],
                     "condition": it["cond"], "label": it["label"], "label_source": it["source"],
                     "y_acc": y_acc, "y_pool": binary(it["label_pool"]),
                     "later_supported": (binary(it["label_traj"]) == 0) if y_acc == 1 else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from scopeverify.llm import VLLMText
    judge = VLLMText(cfg["label_judge"], max_tokens=256, vllm_kwargs=cfg.get("vllm"))
    rows = build_rows(cfg, args.split, lambda prompts, mt: judge(prompts, max_tokens=mt))
    save_jsonl(rows, f"outputs/{args.split}_claims_labeled.jsonl")
    n_c = sum(r["label_source"] == "construction" for r in rows)
    print(f"{len(rows)} labelled claims ({n_c} by construction)")


if __name__ == "__main__":   # required: vLLM starts worker processes with "spawn"
    main()
