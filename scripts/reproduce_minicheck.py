"""Base-paper reproduction: MiniCheck balanced accuracy on LLM-AggreFact.
Compare the printed numbers with the values reported in the MiniCheck paper.
Usage: python scripts/reproduce_minicheck.py --model roberta-large --split test"""
import argparse
import json

import numpy as np
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score

from scopeverify.verifiers import MiniCheckVerifier

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="roberta-large")
ap.add_argument("--split", default="test")
ap.add_argument("--threshold", type=float, default=0.5)
args = ap.parse_args()

ds = load_dataset("lytang/LLM-AggreFact", split=args.split)   # check the dataset card
v = MiniCheckVerifier(args.model)
rows = {}
for name in sorted(set(ds["dataset"])):
    sub = ds.filter(lambda x: x["dataset"] == name)
    probs = np.array(v.score_pairs(list(sub["claim"]), list(sub["doc"])))
    bacc = balanced_accuracy_score(sub["label"], (probs >= args.threshold).astype(int))
    rows[name] = round(100 * bacc, 1)
    print(f"{name:25s} BAcc = {rows[name]:.1f}  (n={len(sub)})")
rows["AVERAGE"] = round(float(np.mean(list(rows.values()))), 1)
print("average BAcc:", rows["AVERAGE"])
json.dump(rows, open(f"outputs/minicheck_{args.model}_repro.json", "w"), indent=2)
