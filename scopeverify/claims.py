"""Step segmentation, claim filtering and decontextualisation (paper Sec. III-A).

`llm_fn` is any callable mapping a list of prompts to a list of strings, e.g.
a thin wrapper around vLLM. Validate both prompts on ~50 hand-checked steps.
"""
import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)

FILTER_PROMPT = (
    "Does the following text assert at least one fact about the world (not only a plan, "
    "a question, or a statement of what to search next)? Answer CLAIM or NOT-A-CLAIM.\n\n"
    "Text: {text}\nAnswer:"
)
DECONTEXT_PROMPT = (
    "Rewrite the LAST TEXT so that it can be understood on its own. Replace pronouns and "
    "references (e.g. 'he', 'the film', '#1') with the entities they refer to, using ONLY "
    "the question and the earlier steps. Keep only factual assertions; drop plans and "
    "questions. Do not add any new information.\n\nQuestion: {question}\n"
    "Earlier steps:\n{history}\n\nLAST TEXT: {text}\n\nRewritten:"
)


def step_text(step):
    m = THINK_RE.search(step["text"])
    body = m.group(1) if m else re.split(r"<search>|<answer>", step["text"])[0]
    return " ".join(body.split())


def heuristic_is_claim(text):
    t = text.strip().lower()
    if not t or t.endswith("?"):
        return False
    return not re.match(r"^(i need|i should|let me|i will|i'll|next,? i|to answer)", t)


def build_claims(trace, llm_fn=None):
    """Return a list of claim dicts for one trace (answer step included)."""
    texts = [step_text(s) for s in trace["steps"]]
    claims = []
    for i, (s, txt) in enumerate(zip(trace["steps"], texts)):
        if not txt:
            continue
        if llm_fn is None:
            keep, dec = heuristic_is_claim(txt), txt
        else:
            keep = "NOT" not in llm_fn([FILTER_PROMPT.format(text=txt)])[0].upper()
            dec = txt
            if keep:
                hist = "\n".join(texts[:i]) or "(none)"
                dec = llm_fn([DECONTEXT_PROMPT.format(question=trace["question"],
                                                      history=hist, text=txt)])[0].strip()
        if keep:
            claims.append({"qid": trace["id"], "t": s["t"], "raw": txt, "claim": dec})
    return claims
