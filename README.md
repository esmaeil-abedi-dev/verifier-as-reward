# Learnable Authorization: Verifier-as-Reward

Code for the paper *"Learnable Authorization: A Verifier-as-Reward Benchmark
and Method for AI-Agent Authority."*

**Thesis.** Whether an agent action is authorized — given the delegation
chain that led to it — is decidable by a deterministic verifier. That
verdict can therefore serve simultaneously as (a) a benchmark label and
(b) a training reward. This repository contains the verifier, a labeled
trace benchmark generated from it, an evaluation harness for frontier
models, and a training harness that uses the verifier as the reward signal.

Everything runs on a CPU-only machine with no network access (the only
exception: pointing the training harness at a real Hugging Face model
downloads its weights). Install dependencies with
`pip install -r requirements.txt` — components 1–2 need only the standard
library (+ numpy); component 3 imports `torch` and `transformers` even for
its offline smoke test.

## The four components

```
authority_verifier.py      the ground truth: deterministic authorization
        │                  verifier over attenuated delegation chains
        │                  (verify / label_action)
        ▼
trace_benchmark.py         generates labeled execution traces across 9
        │                  scenario classes; every label is a verify() call
        │                  → benchmark_train.jsonl, benchmark_test.jsonl,
        │                    DATASHEET.md
        ├──────────────────────────────┐
        ▼                              ▼
eval_harness.py               train_verifier_reward.py
proof-of-life: can a model    REINFORCE loop with label_action() as the
judge authorization? metrics  reward (+1 match / -1 mismatch); CPU smoke
vs. verifier labels           test by default, scales to a GPU model
→ proofoflife_results.json    → training_log.jsonl
```

- **`authority_verifier.py`** — `Grant` / `Scope` / `Delegation` /
  `Action` / `RootAuthority`; `verify(action, chain, root) -> Verdict`
  checks chain structure, per-hop activity (issue/expiry/revocation at the
  action's logical time), per-hop attenuation (no hop may widen its
  parent's scope), and final-scope permission. `label_action` maps the
  verdict to 1/0. **This module is ground truth everywhere; nothing in the
  repo hand-labels.**
- **`trace_benchmark.py`** — seeded generator over 5 domains (email,
  payment, repo, file, db) and 9 scenario classes: `single_delegation`,
  `multi_hop`, `revocation`, `expiry`, `scope_escalation`,
  `resource_violation`, `budget_violation`, `attack_confused_deputy`,
  `chain_structure` (broken links, wrong root, wrong acting agent,
  pre-issue actions). Decoy grants and inert revocation/expiry timestamps
  are mixed in so surface-cue shortcuts misfire. Corpus is ~50/50
  authorized/unauthorized and split 80/20 train/test at the trace level
  (class-stratified, no leakage). Schema and limitations are documented in
  the generated `DATASHEET.md`.
- **`eval_harness.py`** — renders each test action and its delegation
  chain as a natural-language prompt, asks a backend for
  AUTHORIZED/UNAUTHORIZED, and scores it against the verifier label.
  Backends are any `answer(prompt) -> str` callable; built-ins (no
  network): `always_authorized`, `random`, a deliberately shallow
  `heuristic` keyword/number shortcut (the pattern-matching floor a real
  model must beat), and a verifier-backed `oracle` upper bound. Headline
  metric: **false-authorize rate on the attack/violation classes** — the
  dangerous error.
- **`train_verifier_reward.py`** — trains a causal LM to decide
  AUTHORIZE/REFUSE with the verifier verdict as reward (+1 correct
  authorize/refuse, −1 false-authorize or false-refuse) via REINFORCE
  with a running baseline. `--model tiny` (default) is an offline
  random-weight byte-level GPT-2: the CPU smoke test. Any HF causal LM
  name scales it up on a GPU. Logs mean reward and, on a seeded shuffled
  held-out subset, accuracy + violation rate + false-refuse rate vs.
  cumulative training examples to `training_log.jsonl`. Read violation
  rate and false-refuse rate together: a policy that collapses to
  always-refuse zeroes the former while the latter goes to 1.

## Run everything (acceptance order)

```bash
# 0. verifier tests (ground truth must be green first)
PYTHONPATH=. python3 test_authority_verifier.py

# 1. generate the benchmark + its tests
PYTHONPATH=. python3 trace_benchmark.py --seed 7 --traces-per-class 25
PYTHONPATH=. python3 test_trace_benchmark.py

# 2. proof-of-life eval on the built-in baselines + its tests
PYTHONPATH=. python3 eval_harness.py --test-file benchmark_test.jsonl \
    --out proofoflife_results.json
PYTHONPATH=. python3 test_eval_harness.py

# 3. training smoke test (CPU, ~1 min) + its tests
PYTHONPATH=. python3 train_verifier_reward.py
PYTHONPATH=. python3 test_train_verifier_reward.py
```

Artifacts produced: `benchmark_train.jsonl`, `benchmark_test.jsonl`,
`DATASHEET.md`, `proofoflife_results.json`, `training_log.jsonl`.

Reproducibility: all RNGs are seeded; `trace_benchmark.py` with the same
`--seed`/`--traces-per-class` regenerates byte-identical files, and the
training smoke run is deterministic on CPU under a fixed `--seed`.

## Wiring in a real model

**OpenRouter model ladder** (the paper's proof-of-life run). Put the key
in the environment or a `.env` file in the repo root (gitignored — never
commit it):

```bash
# .env
OPENROUTER_API_KEY=sk-or-...
# optional: one *_MODEL var per ladder slot (any names ending in _MODEL;
# they replace the built-in default ladder, ordered by variable name)
SMALL_MODEL=meta-llama/llama-3.1-8b-instruct
MID_MODEL=meta-llama/llama-3.3-70b-instruct
FLASH_MODEL=google/gemini-2.5-flash
FRONTIER_MODEL=anthropic/claude-sonnet-4.5
REASONING_MODEL=deepseek/deepseek-r1
```

```bash
PYTHONPATH=. python3 eval_harness.py --ladder
```

This runs the baselines plus every ladder model (temperature 0, retries
with backoff, per-call timeout) over the same prompts and parsing as the
baselines, prints each model's summary as it finishes, and writes all
entries into `proofoflife_results.json` (saved incrementally, so a crash
mid-ladder keeps finished models). A call that still fails after retries
is recorded as a parse failure — the safe, non-authorizing outcome.
Individual models can also be named directly:
`--backends heuristic openrouter:deepseek/deepseek-r1`.

**Any other API.** From your own script:

```python
from eval_harness import run_eval, print_summary
from trace_benchmark import load_traces

def my_model(prompt: str) -> str:
    ...  # call your API; return text containing AUTHORIZED/UNAUTHORIZED

out = run_eval(my_model, load_traces("benchmark_test.jsonl"))
print_summary("my-model", out["metrics"])
```

**Training objectives and data discipline.** The harness supports sampled
REINFORCE (default; high variance — collapses into blanket policies at
small scale, a documented baseline result) and `--exact-pg`, the
closed-form expected-reward objective: with two decisions and the verifier
able to price both, E[r] is computable exactly, removing the sampling
estimator entirely. For scaled-up training, `make_expanded_train.py`
generates an expanded train corpus and a separate validation corpus from
fresh seeds, deduplicated against the committed `benchmark_test.jsonl` at
the (root, chain, action) decision-context level; the committed test set
is reserved for one final report. Two caveats to disclose alongside
results: training/validation curves use the compact prompt while final
checkpoint evaluation uses the natural-language ladder prompts (the two
scales differ), and the 80-action test set carries a ~±10pp 95% CI on
accuracy.

**Training at scale** (GPU):

```bash
PYTHONPATH=. python3 train_verifier_reward.py \
    --model gpt2 --steps 500 --batch-size 16 --lr 1e-5 --eval-every 25
```

Without a CUDA GPU the harness runs the smoke test and prints a
"READY FOR GPU" message with the scale-up command. The reward path
(`load_examples`, `reward_for_decision`, `candidate_logprobs`) is modular
and drops directly into a TRL/PPO trainer if you prefer PPO over
REINFORCE.
