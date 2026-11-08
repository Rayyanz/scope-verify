import random

import numpy as np
import pandas as pd

from scopeverify.cascade import calibrate_band
from scopeverify.labels import binary, construct_label
from scopeverify.metrics import auroc, recall_at_retention, tune_threshold
from scopeverify.perturb import build_substitute_bank, perturb, resolve_subquestion
from scopeverify.scopes import build_scopes, retrieved_before
from scopeverify.stats import cluster_bootstrap, holm
from scopeverify.text import has, normalize

REC = {
    "id": "2hop__1", "question": "Where was the director of Inception born?", "answer": "London",
    "pool": [
        {"idx": 0, "title": "Inception", "text": "Inception is a 2010 film directed by Christopher Nolan.", "is_supporting": True},
        {"idx": 1, "title": "Christopher Nolan", "text": "Christopher Nolan was born in London.", "is_supporting": True},
        {"idx": 2, "title": "Tenet", "text": "Tenet is a 2020 film.", "is_supporting": False},
    ],
    "hops": [{"question": "Who directed Inception?", "answer": "Christopher Nolan", "support_idx": 0},
             {"question": "Where was #1 born?", "answer": "London", "support_idx": 1}],
}
OTHER = {"id": "x", "question": "q", "answer": "Denis Villeneuve", "pool": [],
         "hops": [{"question": "q", "answer": "Denis Villeneuve", "support_idx": 0}]}
STEPS = [{"t": 1, "retrieved": [0, 2]}, {"t": 2, "retrieved": [1]}, {"t": 3, "retrieved": []}]


def test_normalize_and_has():
    assert normalize("The  Film!") == "film"
    assert has("Directed by Christopher Nolan.", "christopher nolan")
    assert not has("Nolanesque style", "Nolan")


def test_resolve_subquestion():
    assert resolve_subquestion(REC["hops"], 1) == "Where was Christopher Nolan born?"


def test_removed_and_counterfactual():
    rng = random.Random(0)
    p, meta = perturb(REC, "removed", rng)
    assert meta["hop"] == 0 and all(x["idx"] != 0 for x in p["pool"])
    bank = build_substitute_bank([REC, OTHER])
    p, meta = perturb(REC, "counterfactual", random.Random(0), bank=bank)
    cf = next(x for x in p["pool"] if x["idx"] == meta["cf_idx"])
    assert meta["substitute"] == "Denis Villeneuve"
    assert has(cf["text"], "Denis Villeneuve") and not has(cf["text"], "Christopher Nolan")


def test_distractor_is_external():
    ext = [{"idx": "X1", "title": "Avatar", "text": "Avatar was directed by James Cameron.", "is_supporting": False}]
    p, meta = perturb(REC, "distractor", random.Random(0), external=ext, distractor_fn=lambda q, c: c[0])
    ids = [x["idx"] for x in p["pool"]]
    assert 0 not in ids and meta["distractor_idx"] in ids and len(ids) == 3


def test_scopes_and_labels():
    sc = build_scopes(STEPS, 2, REC["pool"], filler=[])
    assert [p["idx"] for p in sc["loc"]] == [0, 2]
    assert [p["idx"] for p in sc["acc"]] == [0, 2]
    assert [p["idx"] for p in sc["traj"]] == [0, 2, 1]
    # removed condition: claim states hop answer that is absent from E^acc -> MB
    p, meta = perturb(REC, "removed", random.Random(0))
    steps = [{"t": 1, "retrieved": [2]}, {"t": 2, "retrieved": [1]}]
    sc = build_scopes(steps, 2, p["pool"], filler=[])
    lab = construct_label("Inception was directed by Christopher Nolan.", meta, sc["acc"],
                          retrieved_before(steps, 2))
    assert lab == "MB" and binary(lab) == 1
    # counterfactual: model keeps original entity after seeing edited paragraph -> CC
    bank = build_substitute_bank([REC, OTHER])
    p, meta = perturb(REC, "counterfactual", random.Random(0), bank=bank)
    steps = [{"t": 1, "retrieved": [meta["cf_idx"]]}, {"t": 2, "retrieved": []}]
    sc = build_scopes(steps, 2, p["pool"], filler=[])
    ids = retrieved_before(steps, 2)
    assert construct_label("Inception was directed by Christopher Nolan.", meta, sc["acc"], ids) == "CC"
    assert construct_label("Inception was directed by Denis Villeneuve.", meta, sc["acc"], ids) == "SUP"
    assert construct_label("Tenet is a film.", meta, sc["acc"], ids) is None


def test_pad_scope_matches_length():
    filler = [{"idx": f"F{i}", "title": "f", "text": "word " * 10} for i in range(10)]
    sc = build_scopes(STEPS, 3, REC["pool"], filler)
    ln = lambda ps: sum(len((p["title"] + " " + p["text"]).split()) for p in ps)
    assert ln(sc["pad"]) >= ln(sc["acc"])


def test_metrics():
    s = np.array([0.9, 0.8, 0.85, 0.2, 0.1, 0.95])
    y = np.array([0, 0, 0, 1, 1, 0])
    assert auroc(s, y) == 1.0
    assert recall_at_retention(s, y, 0.95) == 1.0
    assert 0.2 <= tune_threshold(s, y) <= 0.8


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_bootstrap_and_cascade():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    judge = np.clip(1 - y + rng.normal(0, 0.25, n), 0, 1)
    cheap = np.clip(1 - y + rng.normal(0, 0.45, n), 0, 1)
    df = pd.DataFrame({"cluster": np.repeat(np.arange(100), 4), "s": judge, "y": y})
    p, lo, hi, _ = cluster_bootstrap(df, lambda d: auroc(d["s"], d["y"]), n_boot=200, cluster_col="cluster")
    assert lo <= p <= hi
    band = calibrate_band(cheap, judge, y, 1.0, 50.0, delta=0.02)
    assert band is not None and band["escalation_rate"] < 1.0
