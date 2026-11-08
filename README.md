# scope-verify

Code for the paper **"Evidence Scope and Verification Placement in Step-Level Grounding
Verification for Iterative Multi-Hop Retrieval-Augmented Generation."**

The framework evaluates existing step-level grounding verifiers (DeBERTa NLI, AlignScore,
MiniCheck, an LLM judge) on intermediate claims of Search-R1-style traces under:

- **HLEP** – hop-localized evidence perturbation (removed / distractor / counterfactual) with
  labels by construction (paper Eq. 5–7, Table IV, Algorithm 1);
- **scope-controlled verification** – five evidence scopes `loc, acc, traj, pool, pad`
  (Eq. 2–4, Algorithm 2) and the later-supported rate (Eq. 9);
- **a calibrated verification cascade** (Eq. 10–12, Algorithm 3).

## Repository layout

| Path | Contents | Paper |
|---|---|---|
| `scopeverify/data.py` | MuSiQue / 2Wiki loaders, stratified sampling | III-A |
| `scopeverify/retriever.py` | E5 retrieval over a per-question pool | III-C |
| `scopeverify/perturb.py` | HLEP operators | Eq. 5–6, Alg. 1 |
| `scopeverify/generate.py` | Search-R1 generation loop (vLLM) | III-C |
| `scopeverify/claims.py` | step segmentation, filtering, decontextualisation | III-A |
| `scopeverify/labels.py` | label-construction rules, judge prompt | Eq. 7, Table IV |
| `scopeverify/scopes.py` | evidence scopes | Eq. 2 |
| `scopeverify/verifiers.py` | verifier wrappers, cost meter | Eq. 4 |
| `scopeverify/cascade.py` | cascade and band calibration | Eq. 10–12, Alg. 3 |
| `scopeverify/metrics.py` | AUROC, R@95, F1, BAcc, LSR | Table VI |
| `scopeverify/stats.py` | cluster bootstrap, Holm | IV-B |
| `scripts/01–05_*.py` | end-to-end pipeline | IV |
| `scripts/reproduce_minicheck.py` | base-paper reproduction | IV-C |
| `tests/` | unit tests (CPU only) | – |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m spacy download en_core_web_sm
pytest -q                      # 9 tests, CPU only
```

Download MuSiQue-Ans (official repository) into `data/musique/` and fill the `TODO`
fields in `configs/default.yaml` (generator checkpoint, judge models, cascade delta).

## Running

```bash
# 0. base-paper reproduction (MiniCheck on LLM-AggreFact)
python scripts/reproduce_minicheck.py --model roberta-large

# 1. data
python scripts/01_prepare_data.py
# 2. traces for every condition (+ closed-book run for leakage stratification)
python scripts/02_generate_traces.py --split val
python scripts/02_generate_traces.py --split test
python scripts/02_generate_traces.py --split test --closed_book
# 3. claims and labels
python scripts/03_build_claims_and_labels.py --split val
python scripts/03_build_claims_and_labels.py --split test
# 4. verifier scores with cost metering
python scripts/04_score_verifiers.py --split val  --verifiers nli minicheck alignscore llm_judge --judge_subsample 1.0
python scripts/04_score_verifiers.py --split test --verifiers nli minicheck alignscore llm_judge
# Search-R1 sanity reproduction (EM on clean traces)
python scripts/eval_em.py outputs/test_clean_traces.jsonl
# 5. analysis (delta fixed after the pilot, before looking at test results)
python scripts/05_analyze.py --split test --val_split val --delta <DELTA> --n_boot 10000
```

Outputs are written to `outputs/` (JSONL/CSV, keyed by question id and step) and
`figs/pareto.pdf`. Run the whole pipeline on ~100 validation questions first (pilot).

## Notes

- Step `t` is written after observing `R_1..R_{t-1}`; `loc = R_{t-1}`, `acc = R_1 ∪ … ∪ R_{t-1}`.
- The label judge must differ from the LLM judge evaluated as a verifier.
- The bootstrap is O(n_boot × rows); use `--n_boot 1000` while developing.
- Check the APIs of MiniCheck, AlignScore and vLLM against their current READMEs.
