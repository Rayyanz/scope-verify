"""Apply HLEP conditions and generate Search-R1 traces.
Usage: python scripts/02_generate_traces.py --split test [--closed_book]"""
import argparse
import random

import yaml

from scopeverify.data import load_jsonl, save_jsonl
from scopeverify.generate import generate_traces
from scopeverify.perturb import build_substitute_bank, perturb
from scopeverify.retriever import PoolRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--closed_book", action="store_true")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
g = cfg["generator"]

recs = load_jsonl(f"outputs/{args.split}.jsonl")
external = load_jsonl("outputs/external_paragraphs.jsonl")
retr = PoolRetriever(cfg["retriever"])
try:                                          # spaCy typing: pip install spacy; python -m spacy download en_core_web_sm
    import spacy
    ner = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    ner = None
    print("WARNING: spaCy not available; substitutes are typed only as year/number/entity")
bank = build_substitute_bank(load_jsonl("outputs/val.jsonl") + recs, ner=ner)


def distractor_fn(subq, candidates):
    return retr.search(subq, candidates, k=1)[0]


conditions = ["clean"] if args.closed_book else cfg["conditions"]
for cond in conditions:
    rng = random.Random(cfg["seed"])          # same hop choice across conditions
    pert, metas = [], []
    for r in recs:
        p, meta = perturb(r, cond, rng, bank=bank, external=external, distractor_fn=distractor_fn)
        if p is not None:
            pert.append(p)
            metas.append({"id": r["id"], **meta, "pool": p["pool"]})
    traces = generate_traces(pert, g["model_id"], retr, k=g["top_k"], max_turns=g["max_turns"],
                             max_new_tokens=g["max_new_tokens"], closed_book=args.closed_book)
    tag = "closedbook" if args.closed_book else cond
    save_jsonl(metas, f"outputs/{args.split}_{tag}_meta.jsonl")
    save_jsonl(traces, f"outputs/{args.split}_{tag}_traces.jsonl")
    print(f"{cond}: {len(pert)}/{len(recs)} questions perturbed and generated")
