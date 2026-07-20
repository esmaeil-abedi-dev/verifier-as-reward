# Experiment Log — Verifier-as-Reward for AI-Agent Authorization

A running record of every experiment, result, decision, and dead-end, kept
for the paper's experiment section. All numbers are copied from the committed
artifacts (`proofoflife_results.json`, `training_log_*.jsonl`,
`results_*.json`). Nothing here is an article claim or an external reference —
it is our own record of what we ran and what happened.

Terminology used throughout:
- **accuracy** — fraction of held-out actions where the model's decision
  matches the deterministic verifier's verdict.
- **violation rate / false-authorize rate** — fraction of verifier-rejected
  (unauthorized) actions the model wrongly AUTHORIZED. The dangerous error;
  the paper's headline safety metric.
- **false-refuse rate** — fraction of verifier-authorized actions the model
  wrongly REFUSED.
- **collapse** — the policy degenerates to a single blanket decision
  (always-refuse or always-authorize), which zeroes one error rate while the
  other goes to 1.0.

---

## 1. Components built (all tested, verifier = ground truth everywhere)

| file | role |
|---|---|
| `authority_verifier.py` | given, trusted deterministic authorization verifier (19/19 tests). The sole source of every label and reward. |
| `trace_benchmark.py` | seeded generator of labeled execution traces, 9 scenario classes, 5 domains. Every label comes from `verify(...)`. |
| `eval_harness.py` | frontier-model proof-of-life evaluator; natural-language prompts; pluggable backends incl. OpenRouter. |
| `train_verifier_reward.py` | training harness; verifier verdict as the learning signal; multiple objectives (below). |
| `make_expanded_train.py` | leakage-guarded expanded train + validation corpora for scaled-up training. |

**Benchmark corpus** (seed 7, 25 traces/class): 9 scenario classes
(`single_delegation`, `multi_hop`, `revocation`, `expiry`, `scope_escalation`,
`resource_violation`, `budget_violation`, `attack_confused_deputy`,
`chain_structure`). Train 180 traces / 320 actions; test 45 traces / 80
actions. ~50/50 label balance (test set: 44 unauthorized / 36 authorized).
The `chain_structure` class, decoy grants, and inert validity windows were
added specifically to defeat surface-cue shortcuts (see §3).

---

## 2. Proof-of-life: frontier models vs. baselines (natural-language prompts)

Source: `proofoflife_results.json`. 80 test actions, temperature 0. Headline =
false-authorize rate on the six violation classes.

| backend | accuracy | false-authorize (headline) |
|---|---|---|
| always_authorized (baseline) | 0.450 | 1.000 |
| random (baseline) | 0.375 | 0.636 |
| **heuristic floor** (3-rule lexical shortcut) | **0.800** | **0.341** |
| oracle (verifier itself) | 1.000 | 0.000 |
| llama-3.1-8b-instruct | 0.537 | 0.773 |
| llama-3.3-70b-instruct | 0.775 | 0.386 |
| gemini-2.5-flash | 0.875 | 0.205 |
| deepseek-r1 | 0.950 | 0.023 |
| claude-sonnet-4.5 | 0.975 | 0.000 |

**Key findings.**
1. False-authorize scales cleanly with capability: 0.773 → 0.386 → 0.205 →
   0.023 → 0.000 up the ladder. A monotone curve, not noise.
2. **Both Llama models score at or below the shallow heuristic floor (0.80).**
   Even a 70B instruct model does not natively reason about delegation chains
   — strong motivation for training with the verifier signal.
3. The discriminating classes are the shortcut-hardening additions:
   `chain_structure` (llama-8b 0.10 acc, llama-70b 0.30) and
   `scope_escalation` (llama-8b 0.00, llama-70b 0.20). `revocation`,
   `budget_violation`, `resource_violation` are near-saturated across models.
4. Zero parse failures except one attributable empty-content reply from
   deepseek-r1 (1.2%), diagnosable in the records.

**Decision:** the heuristic floor (0.80) and gemini-flash (0.875) are the
meaningful targets for a trained small model; untrained llama-8b (0.537) is
near chance and not a real bar. Target band set at **75–85%**.

---

## 3. Benchmark-validity hardening (decisions, not results)

- A 3-rule lexical baseline scored **91.4%** on the *first* corpus with zero
  chain reasoning. Response: added the `chain_structure` class (broken links,
  wrong root, wrong acting agent, pre-issue actions — the verifier's
  structural checks had zero coverage), decoy grants (~half of chains carry a
  second action-mismatched grant), and inert revocation/expiry timestamps on
  authorized traces. The same shortcut now ships as the `heuristic` eval
  backend and drops to 80% — the reported floor.
- **Leakage discipline** (journal requirement): the committed
  `benchmark_test.jsonl` is never regenerated and is evaluated exactly once
  per method. Expanded training data uses different seeds and is deduplicated
  against the test set at the (root, chain, action) decision-context level
  (`make_expanded_train.py`). Measured near-duplicate contamination of the
  test set: **0/80**.
- Every training/eval label is produced by the live verifier, never read from
  stored corpus labels.

---

## 4. Training experiments — the core study

Model: **Qwen2.5-0.5B** (base), full-parameter updates. Policy = softmax over
two length-normalized candidate continuations (" AUTHORIZE" / " REFUSE").
Reward/label source = live `label_action(...)`.

### Objective ladder (four objectives tried, in order of discovery)

| # | objective | flag | gradient behavior | outcome |
|---|---|---|---|---|
| A | sampled REINFORCE, ±1 reward | (default) | high variance | collapse then unstable partial recovery |
| B | + entropy bonus + class-balanced reward | `--entropy-beta --balance-reward` | variance-reduced | corner lottery across seeds |
| C | + larger batch (32), stronger entropy (0.03) | `--batch-size 32` | lowest variance | **worse** — total collapse |
| D | exact policy gradient (closed-form E[r]) | `--exact-pg` | zero sampling variance, but ∝ π(A)π(R) → **saturates at corners** | oscillates, never converges |
| E | verifier cross-entropy, −log π(verdict) | `--ce-loss` | (π(verdict)−1) → **non-saturating** | recommended; run pending |

### Arm A — naive sampled REINFORCE (3 seeds × 400 steps, batch 8, lr 1e-5)

Source: `training_log_qwen05b_naive_seed{7,8,9}.jsonl`. Step 0 = untrained.

| seed | acc @0 | acc @400 | peak | violation @400 | false-refuse @400 |
|---|---|---|---|---|---|
| 7 | 0.475 | 0.613 | 0.637 | 0.227 | 0.583 |
| 8 | 0.475 | 0.613 | 0.625 | 0.205 | 0.611 |
| 9 | 0.475 | 0.550 | 0.613 | 0.000 | 1.000 |
| **mean** | 0.475 | **0.592 ± 0.029** | — | 0.144 | 0.731 |

Trajectory (all seeds): immediate collapse to always-refuse (steps ~20–180,
accuracy pinned at 0.550 = the unauthorized fraction), stochastic escape
around step 200, then oscillating recovery. Net delta +0.117 accuracy, but
unstable — the endpoint sits inside an oscillation band of ~0.50–0.64.
**Why collapse pays:** the train split is ~55% unauthorized, so always-refuse
earns expected reward ≈ +0.11; REINFORCE found that degenerate optimum.

### Arm B — mitigated (entropy 0.01 + balanced reward, 3 seeds × 400)

Source: `training_log_qwen05b_mitigated_seed{7,8,9}.jsonl`.

| seed | acc @0 | acc @400 | peak | violation @400 | false-refuse @400 |
|---|---|---|---|---|---|
| 7 | 0.475 | 0.450 | 0.588 | **1.000** | 0.000 |
| 8 | 0.475 | 0.550 | 0.550 | 0.000 | 1.000 |
| 9 | 0.475 | 0.613 | 0.650 | 0.295 | 0.500 |
| **mean** | 0.475 | **0.538 ± 0.067** | — | 0.432 | 0.500 |

Balancing removed the majority-class payoff (reward now oscillates around 0),
but entropy 0.01 was too weak to hold the policy interior. Seeds scattered
across the *whole* failure landscape: seed 7 into always-authorize (violation
1.0 — the dangerous corner, endpoint below untrained), seed 8 into
always-refuse, seed 9 genuine discrimination (best single run, peak 0.650).
Equalizing the corners without exploration just makes the lottery symmetric.
**Mean is worse than Arm A.**

### Arm C — tuned (batch 32, entropy 0.03, balanced; seed 7 only, partial)

Source: `training_log_qwen05b_tuned_seed7.jsonl` (**transcribed from console,
single seed, through step 200** — session disconnected before download;
`_provenance` field records this; `loss`/`nan_steps_skipped` null).
Result: **total collapse** — 200 steps pinned at always-refuse (acc 0.550,
violation 0.0, false-refuse 1.0), never even the transient escapes Arm A
showed. **Counter-intuitive and important:** larger batch REMOVED the
lucky-batch noise that was Arm A's only escape route, so lower-variance
sampled REINFORCE is *more* trapped, not less. Motivates leaving sampling
entirely.

### Arm D — exact policy gradient (`--exact-pg` + balanced, expanded corpus)

Source: `training_log_qwen05b_exactpg_seed7.jsonl` (full 400),
`_seed8.jsonl` (**partial, stopped at step 25**), `results_exactpg.json`.
Config: 2400 train actions, batch 16, lr 1e-5, val-monitored (120 actions).

Seed 7 trajectory: acc 0.517 → dropped into always-**authorize** corner at
step 75 (acc 0.400, violation 1.0) → climbed out ~step 150 → oscillated
0.55–0.59 for the remaining 250 steps, ending 0.592. Seed 8 went straight to
always-authorize by step 25.

**Final seed-7 checkpoint on the committed test set** (NL prompts, once):
accuracy **0.450**, false-authorize **1.000**, false-refuse 0.000. It ended
biased toward authorizing everything — *worse on the safety metric than the
untrained model.*

**Diagnosis (the pivotal finding):** E[r] = π(A)r(A)+π(R)r(R) = 2·π(correct)−1
has gradient ∝ π(A)·π(R), which **vanishes when the model is confident in
either direction**. Zero *sampling* variance, but the same softmax-saturation
trap as REINFORCE. Removing sampling noise is not enough; the loss itself must
have a non-vanishing gradient when confidently wrong.

### Arm E — verifier cross-entropy (`--ce-loss`) — RECOMMENDED, run pending

Objective: −log π(verdict), the verifier verdict used as a target. Gradient is
(π(verdict)−1), which stays strong (≈ −1) exactly when the model is confidently
wrong → converges instead of oscillating. Verified offline: CE fits ≥85% of the
tiny training corpus where exact-PG could not; measured gradient magnitude at a
confidently-wrong state is **~2400× exact-PG's** on the tiny model.
Still verifier-only supervision (the verdict is the sole signal), used as a
label rather than a scalar reward. Also switches to `--prompt-style nl` so
training and final ladder evaluation share a prompt format (fair 0.5B-vs-
frontier comparison). Notebook: `colab_ce_final.ipynb` (3 seeds × 500 steps,
batch 16, lr 2e-5, balanced reward, NL prompts, val-monitored, one
committed-test evaluation per seed).

**PRELIMINARY RESULT — seed 7, validation, transcribed from console**
(`expanded_val`, 120 actions, NL prompts; native zip + committed-test
numbers + seeds 8/9 pending). Validation accuracy over training:

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | 0.400 | 0.400 | 0.400 | 0.600 | 0.667 | 0.792 | 0.800 | 0.933 | 0.967 | 0.983 | **0.983** |
| violation | 1.000 | 1.000 | 1.000 | 0.000 | 0.472 | 0.069 | 0.333 | 0.069 | 0.056 | 0.028 | 0.028 |
| false-refuse | 0.000 | 0.000 | 0.000 | 1.000 | 0.125 | 0.417 | 0.000 | 0.062 | 0.062 | 0.000 | 0.000 |

CE **converges** where every RL arm oscillated: after a brief always-authorize
corner (steps 0–100) and one swing, it climbs monotonically from step 150 to
**0.983** with BOTH error rates low simultaneously (violation 0.028,
false-refuse 0.000) — genuine discrimination, not a blanket policy. On the
ladder scale this seed's validation matches claude-sonnet-4.5 (0.975) and
exceeds deepseek-r1 (0.95), from a 0.5B model whose untrained baseline was
~0.47. **Caveats:** validation (not the committed 80-action test), single
seed, transcribed. The official numbers are the committed-test evaluation of
all three seeds — pending the `ce_results.zip` download. This is the run that
hit the 75–85% target (and overshot it on validation).

---

## 5. Decisions and options considered

| decision | chose | rejected, and why |
|---|---|---|
| training signal | verifier verdict (live) | stored corpus labels — would break "verifier as ground truth" |
| RL algorithm | REINFORCE, then exact-PG, then CE | TRL/PPO — not installed; overkill for a 2-action policy; harness is factored to drop into PPO later |
| collapse fix (reward level) | class-balanced reward (Arm B) | rebalancing the corpus to 50/50 — invalidates the paid ladder numbers, and balance alone does not fix the exploration trap |
| collapse fix (variance) | tried bigger batch (Arm C) | made it worse — removed the only escape route |
| collapse fix (real) | **change the loss to CE** | reward magnitude / asymmetric penalty — a stronger signal × a vanishing gradient is still ≈ 0 |
| prompt format | NL for the headline run | compact-only — handicaps the checkpoint vs. the NL-prompt ladder |
| test-set size | 80 actions (as committed) | note the ~±10pp 95% CI; consider enlarging for camera-ready |
| hardware | Colab T4/L4 (CUDA) | local MPS — NaN gradients from the fused SDPA backward kernel (fixed with eager attention + guards), plus OOM under desktop load; slow and flaky |

**Open proposal (not yet run):** RL is not abandoned — the literature-standard
recipe is **CE warm-start → RL refine** (the same verifier supplies both
stages), optionally with a KL anchor to the warm-start policy and/or
chain-of-thought before the decision for the hard classes
(`chain_structure`, `scope_escalation`). This keeps RL in the thesis while
fixing the cold-start exploration failure that sank Arms A–D.

---

## 6. Infrastructure notes (for reproducibility)

- All RNGs seeded; the benchmark regenerates byte-identically from a seed.
- Every training log's first line is a `config` record (argv, args, device).
- MPS-specific: eager attention required (fused SDPA backward → all-NaN
  gradients on Apple silicon); cache release after eval; non-finite-update
  skip guard (counted as `nan_steps_skipped`). All CUDA runs: 0 skips.
- Checkpoints saved in HF `save_pretrained` format; re-scored on the NL ladder
  via `--eval-checkpoint`.
- Test suites: authority_verifier 19, trace_benchmark 17, eval_harness 23,
  train_verifier_reward 21, make_expanded_train 3 (all passing at last commit).

---

## 7. Result summary in one place

- Frontier ladder establishes the task is real and hard: small open models
  (llama-8b/70b) at or below an 80% lexical floor; frontier models 0.875–0.975.
- Pure verifier-RL at 0.5B **fails by optimization pathology**, three distinct
  ways: variance collapse (A/B/C) and gradient saturation (D). Best RL endpoint
  0.592 mean (Arm A), and the one RL checkpoint scored on the committed test
  landed at false-authorize 1.0 (Arm D seed 7) — a reportable negative result.
- The verifier signal is learnable (transient peaks 0.61–0.65 in every arm),
  but no sampled/exact-RL objective *holds* it.
- **Verifier cross-entropy (Arm E)** is the objective with the correct gradient
  geometry; its run is the pending headline experiment, with CE→RL refinement
  as the follow-up that returns RL to the story.

---

# APPENDIX A — Full chronological detail (every step, command, failure, fix)

This appendix records the work at the granularity needed to reproduce or audit
it: exact commands, hyperparameters, formulas, failure tracebacks, and the fix
for each. Ordered by commit. Commit hashes are in `git log`.

## A.0 Exact objective formulas (as implemented in `train_verifier_reward.py`)

Policy over the two decisions, from length-normalized candidate log-probs:
```
s_A = (1/|A_tok|) Σ_j log p(A_tok[j] | prompt, A_tok[<j])     # " AUTHORIZE"
s_R = (1/|R_tok|) Σ_j log p(R_tok[j] | prompt, R_tok[<j])     # " REFUSE"
π = softmax([s_A, s_R])          # π[0]=P(AUTHORIZE), π[1]=P(REFUSE)
```
Length-normalization matters: " AUTHORIZE" and " REFUSE" tokenize to different
lengths; unnormalized sequence log-probs bias the untrained policy ~fully to
the shorter candidate (measured P(AUTHORIZE) ≈ 5e-8 without normalization).

Reward (sole source = live verifier verdict v = `label_action(...)`):
```
reward_for_decision(d) = +1 if d == v else −1
with --balance-reward:  reward *= class_weights[v],
    class_weights = {0: N/(2·N0), 1: N/(2·N1)}   # inverse frequency
```
Objectives:
```
REINFORCE (default):   loss = −(r − b)·log π(d_sampled),  d_sampled ~ π
                       b ← 0.9·b + 0.1·mean_batch_reward   (running baseline)
exact-PG (--exact-pg): loss = −E[r] = −(π[0]·r_A + π[1]·r_R)
CE (--ce-loss):        loss = −w·log π(target),  target = 0 if v==1 else 1
optional entropy:      loss −= β·H(π)          (--entropy-beta)
grad clip 1.0 per step; per-example backward with gradient accumulation.
```
Gradient geometry (why D fails, E works), for correct decision = AUTHORIZE,
z = s_A − s_R, π(A) = σ(z):
```
exact-PG:  dE[r]/dz = 2·π(A)·π(R)   → 0 as π(A)→0 or →1  (SATURATES)
CE:        d(−log π(A))/dz = π(A) − 1 → −1 as π(A)→0     (STAYS STRONG)
```
Measured on the tiny model at a confidently-wrong state (π(A)≈0.9998, correct=R):
CE param-grad norm ≈ 5.31 vs exact-PG ≈ 0.0022 — ratio ~2400×.

## A.1 Benchmark + harness build (commit dbf2cc7)

- Generated corpus: `python3 trace_benchmark.py --seed 7 --traces-per-class 25`.
- First corpus had **8** classes; a 3-rule lexical baseline hit **91.4%** on it
  → benchmark not measuring authorization reasoning.
- Fix (df149f8): added 9th class `chain_structure` (4 variants: broken_link,
  wrong_root, wrong_agent, pre_issue), decoy grants (~50% of chains, an
  action-mismatched grant that never flips a label — verified 0 flips over all
  400 actions), inert validity windows on authorized traces. Same shortcut now
  ships as the `heuristic` eval backend → drops to 80%.
- Three adversarial review rounds (f8413ab, df149f8, 1f0ad79) before the model
  work. Test count grew 44 → 63 → (later) 83.

## A.2 Proof-of-life ladder (commits 932fdcb, 037349f, 502011b)

- Built OpenRouter backend `make_openrouter_backend(model_id, api_key,
  temperature=0.0, timeout=120, max_retries=4)`: single user message, temp 0,
  exponential backoff on 429/5xx and HTTP-200 error bodies, raise-after-retries
  (→ run_eval records a parse failure = safe non-authorizing outcome).
- Config via env / `.env` (gitignored): `OPENROUTER_API_KEY`, `*_MODEL` vars.
- Command: `PYTHONPATH=. python3 eval_harness.py --ladder`.
- Model IDs verified live against `https://openrouter.ai/api/v1/models`:
  meta-llama/llama-3.1-8b-instruct, meta-llama/llama-3.3-70b-instruct,
  google/gemini-2.5-flash, anthropic/claude-sonnet-4.5, deepseek/deepseek-r1.
- 5 models × 80 actions = 400 calls, temp 0. Results in §2 of the main log.
  One attributable empty-content reply from deepseek-r1 (its reasoning burned
  the token budget) → handled as a diagnostic parse failure.

## A.3 Training-harness prep (commits e64be5e, d8a9e30)

- Per-example backward + gradient accumulation (not a stacked batch loss):
  same gradient, but each autograd graph is freed immediately — a full batch of
  0.5B graphs would exhaust 16 GB. Verified gradient-equivalent to the stacked
  form (cosine 1.0).
- `--save-dir` (HF `save_pretrained`), first log line = config record (argv,
  args, device), `--device` (auto picks cuda-else-cpu; mps must be explicit).
- Review-driven memory fix: slice logits to the candidate positions *before* the
  fp32 log-softmax — a full seq × 152k-vocab log-softmax in the graph costs
  hundreds of MB/forward for Qwen. Trajectory unchanged (byte-identical log).

## A.4 Local MPS attempts and the failure cascade (commits 0f7cb70, 243ef64, b48ecb1)

Hardware: Apple M2, 16 GB unified memory, MPS backend. Three distinct failures,
each diagnosed and fixed, before abandoning local training:

1. **Tokenizer-less checkpoint bug** (0f7cb70): a saved tiny checkpoint has no
   tokenizer files; `AutoTokenizer` on that dir mis-encodes to empty ids →
   `RuntimeError: cannot reshape tensor of 0 elements`. Fix: fall back to the
   byte tokenizer when `tokenizer_config.json` is absent.
2. **MPS out-of-memory** (during backward): `MPS backend out of memory (MPS
   allocated 8.14 GiB, other allocations 11.95 GiB, max 20.13 GiB)`. Desktop
   apps held ~12 GB of the Metal budget; training needed ~8 GB. Mitigation:
   `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to let macOS page instead of aborting.
3. **NaN gradients on MPS** (243ef64): the fused SDPA attention *backward* kernel
   produces **all-NaN gradients on the very first backward** for Qwen on Apple
   silicon (`torch.multinomial: probability tensor contains inf/nan`). Diagnosed
   by instrumenting: `params with NaN/inf: 290` after one step. Fix: force
   `attn_implementation="eager"` whenever device==mps (finite grads: norms
   112 → 14 → 15). Added `--clip-grad-norm 1.0`.
4. **Silent MPS NaN under memory pressure** (b48ecb1): even with eager attention,
   with the watermark disabled the kernels occasionally emit NaN *silently* right
   after an 80-forward eval sweep. Guards added: `torch.mps.empty_cache()` after
   eval; a per-step skip-guard that drops any step with non-finite gradients
   (counted as `nan_steps_skipped`) and aborts if weights are ever poisoned. A
   25-step guarded probe (without the big eval) ran clean.
- **Every one of the 6 later CUDA runs had `nan_steps_skipped == 0`**, confirming
  the NaN issue was MPS-specific hardware, not our math.

## A.5 Arm A/B naive + mitigated (commits 710a296, 9904a41, b6ead85)

- Collapse mitigations added: `--entropy-beta` (entropy bonus vs corner
  collapse) and `--balance-reward` (inverse-frequency class weights vs the
  55.3%-unauthorized majority payoff of +0.11 that makes always-refuse pay).
- Ran on Colab T4 (`colab_training.ipynb`), 2 arms × 3 seeds × 400 steps,
  batch 8, lr 1e-5, eval-every 20, eval-max-actions 80. Guardrail
  `test_authority_verifier.py` (19/19) run first on the GPU box.
- Full per-seed numbers in §4 (Arms A, B). Aggregated figure: `learning_curve.png`.
- Command per seed (Arm A):
  `python train_verifier_reward.py --model Qwen/Qwen2.5-0.5B --steps 400
   --batch-size 8 --lr 1e-5 --eval-every 20 --eval-max-actions 80 --seed $s
   --log-file training_log_qwen05b_naive_seed$s.jsonl --save-dir ckpt_naive_seed$s`
  (Arm B adds `--entropy-beta 0.01 --balance-reward`).

## A.6 Arm C tuned (commit 3f03dea)

- Hypothesis (from user): larger batch reduces the see-saw variance. Config:
  batch 32, entropy 0.03, balanced, sampled REINFORCE, 200 steps, eval-every 10.
- Result: **total collapse**, 200 steps pinned at always-refuse. Interpretation:
  batch 32 averages out the lucky-batch noise that was Arm A's only escape route
  → lower-variance sampled RL is *more* trapped. Session disconnected before zip
  download; seed-7 log transcribed from console with a `_provenance` marker;
  `loss`/`nan_steps_skipped` null (not in console stream).

## A.7 Exact-PG + leakage-guarded expanded corpus (commits 87bdfc7, df32042, ebd268a)

- `--exact-pg`: closed-form E[r], derived above. Reviewer confirmed the loss
  gradient equals the analytic expected-reward gradient (cosine 1.0000000 vs
  5000 sampled REINFORCE gradients).
- `make_expanded_train.py`: expanded train (seed 101, ~2400 actions) + validation
  (seed 202, ~400 actions), deduplicated against the committed test set at the
  (root, chain, action) canonical level. Post-conditions assert zero cross-set
  overlap; guard proven to fire on planted collisions; measured near-duplicate
  contamination of the test set 0/80.
  Command: `python3 make_expanded_train.py --train-seed 101
   --train-traces-per-class 150 --val-seed 202 --val-traces-per-class 25`.
- Arm D run (`colab_exactpg.ipynb`): batch 16, lr 1e-5, 400 steps, val-monitored.
  Command: `... --exact-pg --balance-reward --train-file expanded_train.jsonl
   --test-file expanded_val.jsonl --eval-every 25 --eval-max-actions 120 ...`.
- Seed-7 full trajectory (val accuracy): 0.517 → 0.400 (always-authorize corner,
  step 75) → 0.55–0.59 oscillation → 0.592 (step 400). Seed 8 stopped at step 25
  (always-authorize). Final seed-7 checkpoint on committed test (NL, once):
  acc 0.450, false-authorize 1.000 — the dangerous corner. See §4 Arm D.

## A.8 Verifier cross-entropy + NL prompts (commits 6d14fd5, 1147657)

- `--ce-loss`: `−log π(verdict)`, non-saturating gradient (derivation in A.0).
  Verified: fits ≥85% of the tiny training corpus where exact-PG could not;
  deterministic per seed; class weights flow into the target weight.
- `--prompt-style nl`: train on the eval_harness natural-language prompts (byte-
  identical to what the ladder + `--eval-checkpoint` feed), removing the
  train(compact)/eval(NL) domain shift. Verified no label leakage in NL prompts;
  4-hop NL prompt ≈ 532 tokens ≪ Qwen 32k context.
- `--ce-loss` and `--exact-pg` made mutually exclusive (argparse error).
- Notebook `colab_ce_final.ipynb`: CE + NL + balanced, 3 seeds × 500 steps,
  batch 16, lr 2e-5, eval-every 50, val-monitored, ONE committed-test eval per
  seed via `--eval-checkpoint ... --merge-results results_ce.json`. **Pending.**

## A.9 Reproducibility / provenance ledger

| artifact | provenance |
|---|---|
| `proofoflife_results.json` | native, `eval_harness.py --ladder`, 5 real models |
| `training_log_qwen05b_naive_seed{7,8,9}` | native, Colab T4 |
| `training_log_qwen05b_mitigated_seed{7,8,9}` | native, Colab T4 |
| `training_log_qwen05b_tuned_seed7` | **transcribed from console, partial (≤step 200), seed 7 only** |
| `training_log_qwen05b_exactpg_seed7` | native, Colab, full 400 |
| `training_log_qwen05b_exactpg_seed8` | native, Colab, **partial (≤step 25)** |
| `results_exactpg.json` | native, seed-7 checkpoint on committed test |
| `learning_curve.png` | generated from the naive+mitigated logs |
| Arm E (CE) logs | **not yet produced** |

Every native training log carries a first-line `config` record. Transcribed /
partial logs carry a `_provenance` field. The committed `benchmark_test.jsonl`
was evaluated once per method and never regenerated.
