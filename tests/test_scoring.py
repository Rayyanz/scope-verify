import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


class FakeVerifier:
    """Support = 1 if the claim's last word appears in the document."""
    def score_pairs(self, claims, docs, meter=None):
        if meter:
            meter.start(); meter.stop(tokens_in=10 * len(claims))
        return [1.0 if c.split()[-1].strip(".").lower() in d.lower() else 0.0 for c, d in zip(claims, docs)]


def test_score_claims_max_and_cost():
    spec = importlib.util.spec_from_file_location("s04", ROOT / "scripts/04_score_verifiers.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    claims = [{"qid": "q", "t": 2, "condition": "clean", "label": "SUP", "y_acc": 0, "y_pool": 0,
               "claim": "Nolan was born in London."},
              {"qid": "q", "t": 3, "condition": "clean", "label": "UNG", "y_acc": 1, "y_pool": 1,
               "claim": "Nolan studied at UCL."}]
    paras = [{"title": "A", "text": "Inception film."}, {"title": "B", "text": "Born in London."}]
    scopes_of = [{"acc": paras, "loc": []}, {"acc": paras, "loc": paras[:1]}]
    rows = mod.score_claims(FakeVerifier(), "nli", claims, scopes_of, ["acc", "loc"])
    got = {(r["t"], r["scope"]): r["score"] for r in rows}
    assert got[(2, "acc")] == 1.0 and got[(3, "acc")] == 0.0       # max over paragraphs
    assert got[(2, "loc")] == 0.0                                   # empty scope -> 0
    assert sum(r["tokens_in"] for r in rows if r["scope"] == "acc") > 0
