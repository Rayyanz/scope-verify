import importlib.util
import json
import os
import pathlib
import random

from scopeverify.perturb import build_substitute_bank, perturb
from tests.test_core import OTHER, REC

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("s03", ROOT / "scripts/03_build_claims_and_labels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_llm(prompts, max_tokens):
    out = []
    for p in prompts:
        if p.startswith("Does the following text"):
            out.append("NOT-A-CLAIM" if "I need to" in p.split("Text:")[1][:40] else "CLAIM")
        elif p.startswith("Rewrite the LAST TEXT"):
            out.append(p.split("LAST TEXT: ")[1].split("\n")[0])
        else:
            out.append("A")
    return out


def test_build_rows_counterfactual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs")
    bank = build_substitute_bank([REC, OTHER])
    p, meta = perturb(REC, "counterfactual", random.Random(0), bank=bank)
    trace = {"id": REC["id"], "question": REC["question"], "steps": [
        {"t": 1, "text": "<think>I need to find the director.</think><search>q</search>", "retrieved": [meta["cf_idx"]]},
        {"t": 2, "text": "<think>Inception was directed by Christopher Nolan.</think><search>q2</search>", "retrieved": [1]},
        {"t": 3, "text": "<answer> London </answer>", "retrieved": []}]}
    with open("outputs/val_counterfactual_meta.jsonl", "w") as f:
        f.write(json.dumps({"id": REC["id"], **meta, "pool": p["pool"]}) + "\n")
    with open("outputs/val_counterfactual_traces.jsonl", "w") as f:
        f.write(json.dumps(trace) + "\n")
    rows = _load_script().build_rows({"conditions": ["counterfactual"]}, "val", fake_llm)
    assert len(rows) == 1                      # plan step filtered, answer step empty
    r = rows[0]
    assert r["t"] == 2 and r["label"] == "CC" and r["label_source"] == "construction" and r["y_acc"] == 1
    assert r["y_pool"] == 0 and r["later_supported"] is True   # fake judge says A (supported)
