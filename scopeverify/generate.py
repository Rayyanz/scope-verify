"""Search-interleaved trace generation (Search-R1 style) with vLLM.

Step t's text is written after observing R_1..R_{t-1}; R_t is the result of
step t's search. Prompt, <information> block and document format match the
official Search-R1 code (infer.py, search_r1/llm_agent/generation.py).
"""
import re

SEARCH_R1_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as "
    "many times as your want. If you find no further external knowledge needed, you can "
    "directly provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def format_docs(docs):
    return "\n".join(f"Doc {i+1}(Title: {d['title']}) {d['text']}" for i, d in enumerate(docs))


def generate_traces(records, model_id, retriever, k=3, max_turns=5,
                    max_new_tokens=512, closed_book=False, vllm_kwargs=None):
    """Batched generation. Returns one trace dict per record.
    max_turns=5 gives up to 4 searches plus a final answer turn (Search-R1 max_turns=4)."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, seed=13, **(vllm_kwargs or {}))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                        stop=["</search>", "</answer>"], include_stop_str_in_output=True)

    def start(r):
        q = r["question"].strip()
        if not q.endswith("?"):            # as in Search-R1 infer.py
            q += "?"
        content = SEARCH_R1_PROMPT.format(question=q)
        if tok.chat_template:              # apply only if the tokenizer defines one
            return tok.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
        return content

    state = {r["id"]: {"ctx": start(r), "steps": [], "answer": None, "rec": r} for r in records}
    active = list(state)
    for turn in range(1, max_turns + 1):
        if not active:
            break
        outs = llm.generate([state[i]["ctx"] for i in active], sp)
        still = []
        for qid, o in zip(active, outs):
            st, text = state[qid], o.outputs[0].text
            step = {"t": turn, "text": text, "query": None, "retrieved": [],
                    "prompt_tokens": len(o.prompt_token_ids),
                    "completion_tokens": len(o.outputs[0].token_ids)}
            st["ctx"] += text
            ans = ANSWER_RE.search(text)
            q = SEARCH_RE.search(text)
            if ans:
                st["answer"] = ans.group(1).strip()
            elif q and not closed_book and turn < max_turns:
                step["query"] = q.group(1).strip()
                docs = retriever.search(step["query"], st["rec"]["pool"], k=k)
                step["retrieved"] = [d["idx"] for d in docs]
                st["ctx"] += f"\n\n<information>{format_docs(docs)}</information>\n\n"
                still.append(qid)
            st["steps"].append(step)
        active = still
    return [{"id": qid, "question": s["rec"]["question"], "gold_answer": s["rec"]["answer"],
             "pred_answer": s["answer"], "steps": s["steps"]} for qid, s in state.items()]
