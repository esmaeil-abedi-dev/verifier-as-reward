# E1 — Warm-start RL: does RL help *after* CE?

**Question (the reviewer's top ask).** The paper reports that cold-start
policy-gradient RL fails at 0.5B and that the established fix is a supervised
warm start. Does RL refinement *from the CE-trained policy* then add points,
hold steady, or degrade it?

**Answer: it degrades it — catastrophically, and across every variant.**

## Setup

From the released CE checkpoint (`verifier-ce-qwen2.5-0.5b`, ~0.98 held-out),
three RL arms, 3 seeds each, 500 steps, batch 16, lr 2e-5, on
`expanded_train.jsonl`, evaluated once per seed on the committed test and on a
larger fresh set (both natural-language prompts, Wilson CIs). A KL coefficient
was swept on seed 7 (0.02 / 0.1 / 0.5) before the 3-seed runs.

- **W1** warm + sampled REINFORCE
- **W2** warm + KL-REINFORCE (frozen reference = the warm policy)
- **W3** warm + exact expected-reward policy gradient

## Result

| arm | committed test (seeds 7/8/9) | fresh set (n≈1000) |
|---|---|---|
| **CE start (baseline)** | **97.5%** [91.3, 99.3] | **98.3%** [97.3, 99.0] |
| W1 warm+REINFORCE | 55.0 / 55.0 / 55.0% | 54.4% (all seeds) |
| W2 warm+KL-REINFORCE | 55.0 / 55.0 / 57.5% | 54.4 / 54.4 / 56.5% |
| W3 warm+exact-PG | 55.0 / 55.0 / 55.0% | 54.4% (all seeds) |

Every RL arm collapses the 97.5–98.3% warm policy to ~55% — the accuracy of a
**blanket policy** on this label balance. The collapse mode is always-refuse
in 8 of the 9 committed-test runs (false-refuse 100%, false-authorize 0%);
the lone exception, W2 seed 9, collapsed the *other* way to always-authorize
(false-authorize 77–80%). The training curves show the same story: starting
from acc ≈ 0.992 at step 0, accuracy falls within ~25–50 steps and stabilizes
in the collapsed band.

**KL does not rescue it.** The seed-7 sweep: KL=0.02 → 0.642 (drifting toward
always-refuse), KL=0.1 → 0.600 (false-refuse 1.0), KL=0.5 → 0.600 (false-refuse
1.0). Even the strongest anchor tested collapses to always-refuse; a KL large
enough to fully pin the policy would simply freeze it at the warm start (no
refinement). The peak accuracy over every run is 0.992 — the warm start
itself. RL never improves on step 0.

## Reading / what to claim

- **The clean negative result the reviewer wanted:** at 0.5B, RL refinement
  from a strong CE policy does **not** help — sampled REINFORCE, KL-REINFORCE,
  and exact policy gradient all *degrade* it back to blanket-policy collapse.
  A supervised warm start does not make policy-gradient RL viable at this
  scale; it just gives it a better policy to destroy.
- **This closes the RL question and strengthens the paper's thesis.** The arc
  is now complete: cold-start RL fails (variance / gradient-saturation
  pathologies, Arms A–D); warm-start RL fails too (degrades the CE policy,
  even with KL). Therefore the verifier's verdict is effective as a
  **cross-entropy target**, not as a policy-gradient reward, at this scale —
  CE is the ceiling, not a stepping stone to RL.
- **Honest caveat / future work:** we do not claim RL *cannot* help at any
  scale or with any recipe — larger models, PPO with a value head, GRPO with
  group-relative advantages, or a much larger KL (or KL-to-a-frozen-CE with a
  tiny lr) might refine rather than destroy. Our result is specific and
  reproducible: the standard sampled/exact policy-gradient objectives, warm-
  started and KL-regularized at coefficients up to 0.5, degrade a 0.5B CE
  policy. This motivates those alternatives as future work rather than
  claiming a universal impossibility.

## Deliverables
- `results_warmstart.json` (per-arm × per-seed on committed test + fresh set,
  +Wilson CIs; CE start as the baseline row)
- `training_log_warmstart_{klsweep_*,w1_reinforce,w2_kl,w3_exactpg}_seed*.jsonl`
  (native Colab CUDA logs; first line = exact config)
- Harness: `--warm-start-from` + `--kl-coef` (frozen reference) in
  `train_verifier_reward.py`; `colab_warmstart.ipynb`.
