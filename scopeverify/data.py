"""Loading MuSiQue / 2WikiMultihopQA into one record format and sampling.

Record format:
  {"id", "question", "answer",
   "pool": [{"idx", "title", "text", "is_supporting"}],
   "hops": [{"question", "answer", "support_idx"}]}

Check field names against the official files before running.
"""
import json
import random
from collections import defaultdict


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def musique_record(ex):
    pool = [{"idx": p["idx"], "title": p["title"], "text": p["paragraph_text"],
             "is_supporting": bool(p.get("is_supporting", False))} for p in ex["paragraphs"]]
    hops = [{"question": h["question"], "answer": h["answer"],
             "support_idx": h["paragraph_support_idx"]} for h in ex["question_decomposition"]]
    return {"id": ex["id"], "question": ex["question"], "answer": ex["answer"],
            "pool": pool, "hops": hops}


def load_musique(path, answerable_only=True):
    rows = load_jsonl(path)
    if answerable_only:
        rows = [r for r in rows if r.get("answerable", True)]
    return [musique_record(r) for r in rows]


def twowiki_record(ex):
    """2Wiki: context = [[title, [sentences]]], evidences = [[s, r, o]]."""
    pool = [{"idx": i, "title": t, "text": " ".join(sents), "is_supporting": False}
            for i, (t, sents) in enumerate(ex["context"])]
    by_title = {p["title"]: p["idx"] for p in pool}
    hops = []
    for s, r, o in ex.get("evidences", []):
        idx = by_title.get(s)
        if idx is not None:
            pool[idx]["is_supporting"] = True
        hops.append({"question": f"{s} {r}?", "answer": o, "support_idx": idx})
    return {"id": ex["_id"], "question": ex["question"], "answer": ex["answer"],
            "pool": pool, "hops": hops}


def load_twowiki(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [twowiki_record(r) for r in rows]


def stratified_sample(records, n, seed=13, key=lambda r: len(r["hops"])):
    """Sample about n records with strata proportional to their frequency."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in records:
        strata[key(r)].append(r)
    total = len(records)
    out = []
    for k in sorted(strata):
        group = strata[k]
        m = round(n * len(group) / total)
        out.extend(rng.sample(group, min(m, len(group))))
    rng.shuffle(out)
    return out[:n]
