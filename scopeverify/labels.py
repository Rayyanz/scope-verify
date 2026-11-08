"""Label construction by HLEP (paper Eq. 7, Table IV) and the LLM-judge prompt.

Returned labels: 'MB' (Missing Bridge), 'IE' (Irrelevant Evidence),
'CC' (Contradicted Claim), 'SUP' (supported), or None (defer to judge).
Binary gold: y = 1 (ungrounded) for MB/IE/CC/UNG, y = 0 for SUP.
"""
from .text import has, in_evidence

UNGROUNDED = {"MB", "IE", "CC", "UNG"}

JUDGE_PROMPT = (
    "You are checking whether a claim is supported by evidence.\n"
    "Use ONLY the evidence below, not your own knowledge. A claim is SUPPORTED only if a "
    "typical reader would conclude it is true from the evidence alone (trivial inference "
    "such as synonyms or date arithmetic is allowed). If the claim has several facts, ALL "
    "must be supported.\n\nEvidence:\n{evidence}\n\nClaim: {claim}\n\n"
    "Answer with one letter: A) SUPPORTED  B) CONTRADICTED  C) NOT ENOUGH INFORMATION\nAnswer:"
)


def aligned(claim, meta):
    if not meta or meta.get("hop") is None:
        return False
    return has(claim, meta["answer"]) or (meta.get("substitute") and has(claim, meta["substitute"]))


def construct_label(claim, meta, acc_paragraphs, acc_ids):
    """Rule table Lambda. acc_paragraphs: E^acc_t; acc_ids: their idx values."""
    if not aligned(claim, meta):
        return None
    m, a = meta["condition"], meta["answer"]
    states_a = has(claim, a)
    a_in_acc = in_evidence(acc_paragraphs, a)
    if m in ("removed", "distractor") and states_a and not a_in_acc:
        if m == "distractor" and meta.get("distractor_idx") in acc_ids:
            return "IE"
        return "MB"
    if m == "counterfactual":
        sub = meta["substitute"]
        cf_seen = meta["cf_idx"] in acc_ids
        if states_a and not has(claim, sub) and cf_seen and not a_in_acc:
            return "CC"
        if has(claim, sub) and cf_seen:
            return "SUP"
    return None


def format_evidence(paragraphs):
    return "\n".join(f"[{i+1}] {p['title']}: {p['text']}" for i, p in enumerate(paragraphs)) or "(none)"


def parse_judge(text):
    t = text.strip().upper()[:1]
    return {"A": "SUP", "B": "UNG", "C": "UNG"}.get(t)


def binary(label):
    return None if label is None else int(label in UNGROUNDED)
