"""Minimal vLLM text wrapper used for claim filtering, decontextualisation and
label judging. Use a different model family from the generator."""


class VLLMText:
    def __init__(self, model_id, tp=1, max_tokens=256):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_id, tensor_parallel_size=tp, seed=13)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def __call__(self, prompts, max_tokens=None):
        from vllm import SamplingParams
        sp = self.sp if max_tokens is None else SamplingParams(temperature=0.0, max_tokens=max_tokens)
        chats = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        return [o.outputs[0].text for o in self.llm.generate(chats, sp)]
