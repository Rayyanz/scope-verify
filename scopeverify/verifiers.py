"""Verifier wrappers with a common interface and cost metering (paper Eq. 4).

Every verifier implements score_pairs(claims, docs) -> list[float] in [0, 1]
(higher = more supported). Paragraph-level max aggregation over a scope is done
by `score_scope`. Check each package's README for its current API.
"""
import time
from dataclasses import dataclass, field


@dataclass
class CostMeter:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    _t0: float = field(default=0.0, repr=False)

    def start(self):
        _sync()
        self._t0 = time.perf_counter()

    def stop(self, calls=0, tokens_in=0, tokens_out=0):
        _sync()
        self.seconds += time.perf_counter() - self._t0
        self.calls += calls
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class NLIVerifier:
    name = "nli"

    def __init__(self, model_id="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                 device="cuda", batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device, self.bs = torch, device, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent = l2i["entailment"]

    def score_pairs(self, claims, docs, meter=None):
        out = []
        for i in range(0, len(claims), self.bs):
            enc = self.tok(docs[i:i + self.bs], claims[i:i + self.bs], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            if meter:
                meter.start()
            with self.torch.no_grad():
                p = self.model(**enc).logits.softmax(-1)[:, self.ent]
            if meter:
                meter.stop(calls=0, tokens_in=int(enc["attention_mask"].sum()))
            out.extend(p.tolist())
        return out


class AlignScoreVerifier:
    name = "alignscore"

    def __init__(self, ckpt_path, device="cuda:0", batch_size=32):
        from alignscore import AlignScore
        self.scorer = AlignScore(model="roberta-large", batch_size=batch_size, device=device,
                                 ckpt_path=ckpt_path, evaluation_mode="nli_sp")

    def score_pairs(self, claims, docs, meter=None):
        if meter:
            meter.start()
        s = self.scorer.score(contexts=docs, claims=claims)
        if meter:
            meter.stop()
        return [float(x) for x in s]


class MiniCheckVerifier:
    name = "minicheck"

    def __init__(self, model_name="roberta-large", cache_dir="./ckpts"):
        from minicheck.minicheck import MiniCheck
        self.scorer = MiniCheck(model_name=model_name, cache_dir=cache_dir)

    def score_pairs(self, claims, docs, meter=None):
        if meter:
            meter.start()
        _, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims)
        if meter:
            meter.stop()
        return [float(x) for x in raw_prob]


class LLMJudgeVerifier:
    """Score = P(A) / (P(A)+P(B)+P(C)) from first-token log-probabilities."""
    name = "llm_judge"

    def __init__(self, model_id, vllm_kwargs=None):
        from vllm import LLM, SamplingParams
        from .labels import JUDGE_PROMPT
        self.llm = LLM(model=model_id, seed=13, **(vllm_kwargs or {}))
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        self.prompt = JUDGE_PROMPT

    def score_pairs(self, claims, docs, meter=None):
        import math
        prompts = [self.tok.apply_chat_template(
            [{"role": "user", "content": self.prompt.format(evidence=d, claim=c)}],
            tokenize=False, add_generation_prompt=True) for c, d in zip(claims, docs)]
        if meter:
            meter.start()
        outs = self.llm.generate(prompts, self.sp)
        if meter:
            meter.stop(calls=len(prompts), tokens_in=sum(len(o.prompt_token_ids) for o in outs),
                       tokens_out=len(outs))
        scores = []
        for o in outs:
            p = {"A": 0.0, "B": 0.0, "C": 0.0}
            for lp in o.outputs[0].logprobs[0].values():
                key = (lp.decoded_token or "").strip().upper()
                if key in p:
                    p[key] += math.exp(lp.logprob)
            z = sum(p.values())
            scores.append(p["A"] / z if z > 0 else 0.5)
        return scores


def score_scope(verifier, claim, paragraphs, chunk="paragraph", meter=None):
    """Eq. 4: max over chunks. chunk='paragraph' (encoders) or 'all' (LLM judge)."""
    from .labels import format_evidence
    if not paragraphs:
        return 0.0
    if chunk == "all":
        return verifier.score_pairs([claim], [format_evidence(paragraphs)], meter)[0]
    docs = [p["title"] + ". " + p["text"] for p in paragraphs]
    return max(verifier.score_pairs([claim] * len(docs), docs, meter))
