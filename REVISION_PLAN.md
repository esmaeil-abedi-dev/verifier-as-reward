# Revision Plan — five major-revision experiments

Plan only; no code yet. Addresses the four open reviewer items (E1–E5) in the
existing repo. **Hard constraints (unchanged from the whole project):**
`authority_verifier.py` and `trace_benchmark.py` are frozen; every label/number
comes from `label_action`/`verify`, never hand-written; committed test evaluated
once per seed; all tuning on val/fresh sets; 95% Wilson CIs on every
accuracy/false-authorize figure; weights → HF Hub, never git; keep all 95 tests
green and add tests for new code.

## What already exists (so we don't rebuild it)

- `eval_harness.run_eval` already returns **per-action records**
  (`trace_id, scenario_class, t, label, prediction, raw_reply, error`) and
  `--keep-records` persists them. → E2's dump is a thin add (needs the rendered
  `prompt` and `failing_hop_true` fields), not a new evaluator.
- `train_verifier_reward.py` is modular with `--ce-loss`, `--exact-pg`,
  `--balance-reward`, `--save-dir`, `--eval-checkpoint`, `--merge-results`,
  `build_model_and_tokenizer` (already loads a local/HF checkpoint dir). →
  E1 adds `--warm-start-from` and `--kl-coef`; the loading path already exists.
- No Wilson-CI helper and no `--fresh` flag exist. → add a shared
  `stats.wilson_ci()` used by all experiments; "fresh set" = generate with a
  new seed and pass as `--test-file` (no new flag needed).

## Shared prerequisite (do first, one small module + tests)

- **`stats.py`**: `wilson_ci(k, n, z=1.96) -> (lo, hi)` and a helper to attach
  CIs to a metrics dict. Reason it's shared: E1–E5 all report CIs; put it in
  one tested place and have `compute_metrics` optionally include
  `accuracy_ci` / `false_authorize_ci` (n = the action count each was measured
  on). `test_stats.py` checks against known Wilson values and edge cases
  (k=0, k=n, n=0).
- This is a pure add to `eval_harness` output; it does not change any existing
  number, only annotates it.

---

## E1 — Warm-start RL (highest value). New code in `train_verifier_reward.py`.

**Goal:** does RL *after* CE add / hold / degrade the ~0.98 warm policy? Report
Δ vs. the CE start with Wilson CIs. If KL is what keeps RL from wandering, that
is the finding.

**Code changes (train harness):**
1. `--warm-start-from PATH`: load initial weights from a CE checkpoint via the
   existing `build_model_and_tokenizer` dir path instead of the base model;
   keep the same tokenizer. (Loading a checkpoint dir already works — this is
   mostly wiring it into `train()` as the init.)
2. `--kl-coef FLOAT` (default 0.0): hold a **frozen reference policy** (a second
   copy of the warm-start model, `requires_grad_(False)`, `eval()`), and add a
   per-decision KL penalty toward it: `loss += kl_coef * KL(pi || pi_ref)` over
   the two-way decision softmax (A/R), computed on the same forward the policy
   uses. Frozen ref = no grad, one extra forward per example.
3. Memory note: two 0.5B models resident (policy + frozen ref). Fits a T4/L4;
   on a 16 GB Mac only via forward-only, so **train on Colab** (as before).

**Runs (Colab, cloned from GitHub):** warm-start checkpoint = the released
`verifier-ce-qwen2.5-0.5b` (or retrain one CE seed → `ckpt_ce_warmstart/`).
Three arms × 3 seeds (7,8,9), `expanded_train.jsonl`, 500 steps, batch 16,
val-monitored every 25:
- **W1 warm+REINFORCE:** `--warm-start-from ... --balance-reward` (sampled PG).
- **W2 warm+KL-REINFORCE:** W1 + `--kl-coef`; first sweep {0.02, 0.1, 0.5} on
  **seed 7 only** (tune on val), then 3 seeds at the best coef.
- **W3 warm+exact-PG:** `--warm-start-from ... --exact-pg`.

**Eval:** every final checkpoint on `benchmark_test.jsonl` **and** a fresh
generated set (new seed, dedup vs train) via `eval_harness.py`; per-seed
accuracy / false-authorize / false-refuse with Wilson CIs; Δ vs. CE start.

**Deliverables:** `results_warmstart.json` (results_ce.json schema +CIs),
`training_log_warmstart_{arm}_{seed}.jsonl`, `WARMSTART_FINDINGS.md` (Δ + verdict:
add/hold/degrade, and whether KL is load-bearing).

**Tests:** `--warm-start-from` initializes from the checkpoint (weights ≠ base);
frozen ref has no grad and is unchanged after a step; `--kl-coef 0` reproduces
the no-KL loss exactly; KL term ≥ 0 and 0 when policy == ref. Tiny-model CPU
smoke, as with every other objective.

**Risk/È honesty:** starting at 0.98 on an 80-action test (±~10pp CI), small Δ
may be inside noise — that's why we also eval on a large fresh set for a tighter
CI, and report Δ with CIs rather than a point estimate.

---

## E2 — Qualitative / interpretability. New `dump_predictions.py` (thin).

**Goal:** trace-level examples + error-mode table; confirm/refute
"residual error concentrates in `chain_structure`."

**Code:** small script that calls `run_eval(..., keep records)` for a backend
and writes `predictions_<backend>.jsonl` with
`{trace_id, scenario_class, prompt, label, prediction, failing_hop_true,
model_reply_raw}`. Needs two additions to the record path: attach the rendered
`prompt` (already built in `run_eval`) and `failing_hop_true` (from the stored
trace's action). No metric changes.

**Runs (mostly local — forward-only):** CE-trained 0.5B (`local:` backend, runs
on MPS locally); frontier backends (`openrouter:` for sonnet-4.5, deepseek-r1,
llama-8b) need the `.env` key and network. All on `benchmark_test.jsonl`.

**Analysis (offline, from the JSONLs):** three curated sets — CE-right/frontier-
wrong, frontier-right/CE-wrong, CE-still-wrong; misclassification-by-class and
by structural feature (chain depth, decoy present, tight timing window). All
derivable from stored trace fields — no re-running.

**Deliverables:** `predictions_<backend>.jsonl` (×4), `QUALITATIVE_ANALYSIS.md`
(3–5 rendered traces with all verdicts + true failing hop, and the
by-class table). Appendix figure/table.

**Note:** low risk, cheap; the 80-action set means small per-class counts —
report counts not just rates, and pull extra examples from a fresh set if a
class has too few misclassifications to illustrate.

---

## E3 — Noise / robustness surrogate. New `make_noisy_test.py`.

**Goal:** show graceful degradation under surface "messiness," proving the gains
aren't an artifact of clean templated text — **without changing any label.**

**Code (surface-only perturbations of rendered/structural fields that the
verifier does NOT read for its verdict):**
- reworded / ungrammatical `note` text; drop optional descriptive phrasing;
- inject 1–3 irrelevant extraneous tool-call lines into the **rendered prompt**;
- reorder delegation lines only where order is not load-bearing.
- **Critical guard:** after each perturbation, re-run `label_action` on the
  (root, chain, action) and assert the label equals the clean label; **discard
  and count** any trace whose label flipped. Report discards.

Key subtlety to get right: the perturbations must touch only what the *prompt
renderer* / non-semantic fields see, never the (action, resource, amount, t,
scope, timing) the verifier decides on. Delegation reorder is only safe when it
doesn't change delegator→delegatee linkage; the assert is the backstop.

**Runs:** CE-trained 0.5B + 1–2 frontier models on `noisy_test.jsonl` vs the
clean committed test; accuracy / false-authorize with Wilson CIs; clean→noisy Δ.

**Deliverables:** `noisy_test.jsonl` (+discard count), `results_noisy.json`,
`NOISE_FINDINGS.md`.

**Tests:** every emitted noisy trace's label re-verifies and equals its clean
source's label (this is the whole validity claim); discard path exercised;
deterministic from a seed.

---

## E4 — CE data-scaling ablation (if time). No new code.

**Goal:** committed-test accuracy vs. train size — data-efficiency curve.

**Runs:** `--ce-loss` (winning config) on 200 / 400 / 800 / 1600 / 2400 actions,
1 seed each. Build the sub-corpora as class-stratified prefixes of
`expanded_train.jsonl` (a small helper or `head`-style subsampler; keep the
existing 2400 point as the top of the curve). Eval each on committed test with
Wilson CIs.

**Deliverables:** `results_ce_datascaling.json`, learning-curve figure.
Uses existing training + eval; only a corpus-subsampling helper is new.

---

## E5 — Real-trace external validity (high value). New `map_tau_to_chain.py`.

**Goal:** show authorization judgment transfers to **independently-authored**
real tool-call traces (tau2-bench), with our verifier assigning every label.

**Steps:**
1. `pip install datasets`; pull one tau2 set from HF
   (`snorkelai/Tau2-Bench-Airline-With-Code-Agents` or
   `Jarrodbarnes/tau2-sft-v4-dataset`, Apache-2.0). Inspect the tool-call schema
   (tool name, args, actor/role per turn). **This is the one experiment needing
   a network download + license check.**
2. `map_tau_to_chain.py`: each trajectory → our `(root, delegations, action)`:
   - user/system policy → `root` scope (authority granted, e.g. "read/modify
     this user's booking only");
   - assistant→tool calls → `actions` (`action`, `resource`, `amount` from tool
     args);
   - single-principal dataset ⇒ single-hop chain (root → agent); **document
     this as the fidelity limit.**
3. `label_action(action, chain, root)` assigns every label. Never hand-label.
   Save `real_trace_test.jsonl`.
4. **Manual fidelity audit** of ~15 mapped+labeled actions: does the verifier
   verdict match what the tau2 policy implies? Report the agreement rate; if
   lossy, say so numerically.
5. Eval CE-trained 0.5B + sonnet-4.5 + llama-8b on `real_trace_test.jsonl`;
   accuracy / false-authorize with Wilson CIs.

**Claim discipline:** "authorization judgment on independently-authored real
tool-call traces," NOT "real multi-hop delegation logs" (tau2 is single-
principal; attenuation is partly synthetic in the mapping). State the single-hop
limit plainly.

**Deliverables:** `map_tau_to_chain.py`, `real_trace_test.jsonl`
(+discard/agreement counts), `results_realtrace.json`, `REALTRACE_FINDINGS.md`.

**Risks:** dataset schema may not cleanly expose the policy/authority — mapping
may be lossy or require judgment calls (record them); some trajectories may not
map at all (report the map/discard rate). This is the highest-uncertainty
experiment; timebox the mapping and report whatever fraction maps faithfully.

---

## Suggested order, dependencies, and where each runs

1. **`stats.py` + Wilson CIs into `compute_metrics`** (prerequisite for all).
2. **E2** (cheap, mostly local, no train) — validates the prediction-dump path
   and the chain_structure claim early.
3. **E3** (generator-side, local; the label-invariance guard is the core work).
4. **E1** (the headline; needs the two train-harness flags + Colab GPU runs;
   biggest compute).
5. **E5** (external download + mapping; highest uncertainty; timeboxed).
6. **E4** (if time; pure reruns of the winning config at smaller sizes).

**Compute split:** E2/E3/E4-eval and all `local:` forward-only evals run on the
Mac (MPS, fine for inference). E1 and E4 *training* run on Colab (GPU), cloned
from GitHub; user runs `! git push` since my pushes are sandbox-blocked. E5 map
+ label runs locally; its evals are local (CE) + OpenRouter (frontier).

**Per experiment, the review discipline stays:** add `test_*.py`, keep the
suite green, one adversarial review pass on new code before declaring done,
provenance-label transcribed vs native numbers, and hand back result JSONs +
findings `.md` for folding into `paper_C.tex` (I do not edit the .tex).

## Open questions to confirm before building (do not block planning)

- **Warm-start source:** use the *released* seed-9 CE checkpoint (reproducible,
  already public) vs. retrain a fresh CE seed for a clean `--save-dir`? Plan
  assumes the released checkpoint (cheaper, public, reproducible).
- **KL definition:** KL over the two-way decision softmax (A/R) is the natural,
  cheap choice and matches the policy; full token-level KL is heavier and not
  needed for a 2-action policy. Plan assumes decision-level KL.
- **Fresh-set size for E1 tighter CIs:** propose ~600–1000 actions (new seed,
  dedup vs train) so Δ vs. CE has a usable CI.
- **E5 dataset choice** between the two HF mirrors — pick after inspecting which
  exposes tool args + policy most cleanly.
