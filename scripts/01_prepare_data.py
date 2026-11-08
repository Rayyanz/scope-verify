"""Sample test/validation questions and an external paragraph bank.
Usage: python scripts/01_prepare_data.py --config configs/default.yaml"""
import argparse
import random

import yaml

from scopeverify.data import load_musique, save_jsonl, stratified_sample

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
d, seed = cfg["data"], cfg["seed"]

dev = load_musique(f"{d['musique_dir']}/musique_ans_v1.0_dev.jsonl")
train = load_musique(f"{d['musique_dir']}/musique_ans_v1.0_train.jsonl")
test = stratified_sample(dev, d["n_test"], seed)
val = stratified_sample(train, d["n_val"], seed)

used = {r["id"] for r in test + val}
rng = random.Random(seed)
bank = [dict(p, idx=f"X{r['id']}_{p['idx']}") for r in train if r["id"] not in used for p in r["pool"]]
external = rng.sample(bank, min(d["n_external"], len(bank)))

save_jsonl(test, "outputs/test.jsonl")
save_jsonl(val, "outputs/val.jsonl")
save_jsonl(external, "outputs/external_paragraphs.jsonl")
hops = {}
for r in test:
    hops[len(r["hops"])] = hops.get(len(r["hops"]), 0) + 1
print(f"test={len(test)} val={len(val)} external={len(external)} hop distribution={hops}")
