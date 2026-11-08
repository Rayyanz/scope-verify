"""Exact match of generated answers (Search-R1 reproduction check, paper Table VII).
Usage: python scripts/eval_em.py outputs/test_clean_traces.jsonl"""
import sys

from scopeverify.data import load_jsonl
from scopeverify.text import normalize

traces = load_jsonl(sys.argv[1])
em = [normalize(t["pred_answer"] or "") == normalize(t["gold_answer"]) for t in traces]
print(f"EM = {100 * sum(em) / len(em):.1f}  (n={len(em)})")
