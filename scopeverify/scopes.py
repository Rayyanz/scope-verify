"""Evidence scopes (paper Eq. 2): loc, acc, traj, pool, pad.

Step t (1-based) was written after observing R_1..R_{t-1}, where R_j is the
'retrieved' list of step j.
"""


def _tok_len(paragraphs):
    return sum(len((p["title"] + " " + p["text"]).split()) for p in paragraphs)


def build_scopes(trace_steps, t, pool, filler):
    """Return {scope: [paragraph dicts]} for step t.
    pool: perturbed pool of the question; filler: paragraphs from OTHER questions."""
    by_idx = {p["idx"]: p for p in pool}
    R = {s["t"]: s["retrieved"] for s in trace_steps}

    def paras(ids):
        seen, out = set(), []
        for i in ids:
            if i in by_idx and i not in seen:
                seen.add(i)
                out.append(by_idx[i])
        return out

    loc = paras(R.get(t - 1, []))
    acc = paras([i for j in sorted(R) if j < t for i in R[j]])
    traj = paras([i for j in sorted(R) for i in R[j]])
    pad, target = list(loc), _tok_len(acc)
    for f in filler:
        if _tok_len(pad) >= target:
            break
        pad.append(f)
    return {"loc": loc, "acc": acc, "traj": traj, "pool": list(pool), "pad": pad}


def retrieved_before(trace_steps, t):
    """Set of paragraph idx observed before step t (for label construction)."""
    return {i for s in trace_steps if s["t"] < t for i in s["retrieved"]}
