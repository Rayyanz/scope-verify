"""HLEP: hop-localized evidence perturbation (paper Eq. 5-6, Algorithm 1).

Conditions: clean, removed, distractor, counterfactual.
All randomness is controlled by an explicit random.Random instance.
"""
import copy
import random
import re

from .text import has

YEAR = re.compile(r"^\d{4}$")
NUMBER = re.compile(r"^[\d.,]+$")


def answer_type(a: str, ner=None) -> str:
    """Coarse answer type. Pass `ner` (a spaCy Language, e.g. en_core_web_sm) to
    separate PERSON / GPE / ORG / DATE ...; otherwise year / number / entity."""
    a = a.strip()
    if YEAR.match(a):
        return "year"
    if NUMBER.match(a):
        return "number"
    if ner is not None:
        ents = ner(a).ents
        if ents:
            return ents[0].label_
    return "entity"


def resolve_subquestion(hops, k):
    """Replace '#j' references in hop k's question with earlier gold answers."""
    q = hops[k]["question"]
    for j in range(k, 0, -1):
        q = q.replace(f"#{j}", hops[j - 1]["answer"])
    return q


def choose_hop(rec, rng: random.Random):
    """Choose a non-final hop with a known supporting paragraph (bridge hop)."""
    cands = [i for i, h in enumerate(rec["hops"][:-1]) if h["support_idx"] is not None]
    if not cands:
        cands = [i for i, h in enumerate(rec["hops"]) if h["support_idx"] is not None]
    return rng.choice(cands) if cands else None


def replace_answer(paragraph, old, new):
    """p[a -> b]: case-insensitive replacement in title and text."""
    pat = re.compile(re.escape(old), re.IGNORECASE)
    p = dict(paragraph)
    p["title"] = pat.sub(new, p["title"])
    p["text"] = pat.sub(new, p["text"])
    return p


def build_substitute_bank(records, ner=None):
    """Map answer type -> list of answers from all records (for substitutes)."""
    bank = {}
    for r in records:
        for h in r["hops"]:
            bank.setdefault(answer_type(h["answer"], ner), set()).add(h["answer"].strip())
    bank = {t: sorted(v) for t, v in bank.items()}
    bank["_ner"] = ner
    return bank


def pick_substitute(answer, bank, rng, exclude=()):
    """Same-type answer from another question; never an answer of this question."""
    pool = [a for a in bank.get(answer_type(answer, bank.get("_ner")), [])
            if not has(a, answer) and not has(answer, a)
            and not any(has(a, e) or has(e, a) for e in exclude)]
    return rng.choice(pool) if pool else None


def perturb(rec, condition, rng, bank=None, external=None, distractor_fn=None):
    """Return (perturbed_record, meta). `external` is a list of paragraphs from
    other questions; `distractor_fn(subq, candidates) -> paragraph` ranks them
    (e.g. with PoolRetriever.search). Returns meta=None if not applicable."""
    out = copy.deepcopy(rec)
    meta = {"condition": condition, "hop": None}
    if condition == "clean":
        return out, meta
    k = choose_hop(rec, rng)
    if k is None:
        return None, None
    hop = rec["hops"][k]
    gold_idx, a_k = hop["support_idx"], hop["answer"]
    gold = next(p for p in rec["pool"] if p["idx"] == gold_idx)
    rest = [p for p in out["pool"] if p["idx"] != gold_idx]
    meta.update(hop=k, answer=a_k, gold_idx=gold_idx, subq=resolve_subquestion(rec["hops"], k))

    if condition == "removed":
        out["pool"] = rest
    elif condition == "distractor":
        cands = [p for p in (external or []) if not has(p["title"] + " " + p["text"], a_k)]
        if not cands or distractor_fn is None:
            return None, None
        d = dict(distractor_fn(meta["subq"], cands))
        d["idx"], d["is_supporting"] = f"D{gold_idx}", False
        out["pool"] = rest + [d]
        meta["distractor_idx"] = d["idx"]
    elif condition == "counterfactual":
        sub = pick_substitute(a_k, bank or {}, rng, exclude=[h["answer"] for h in rec["hops"]])
        if sub is None:
            return None, None
        cf = replace_answer(gold, a_k, sub)
        if has(cf["title"] + " " + cf["text"], a_k) or not has(cf["text"], sub):
            return None, None  # replacement failed; skip question (Alg. 1 assert)
        cf["idx"] = f"CF{gold_idx}"
        out["pool"] = rest + [cf]
        meta.update(substitute=sub, cf_idx=cf["idx"])
    else:
        raise ValueError(condition)
    return out, meta
