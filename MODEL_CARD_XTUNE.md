---
license: apache-2.0
base_model: esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b
pipeline_tag: text-generation
tags:
  - authorization
  - ai-agent-safety
  - verifier-as-reward
  - consistency-regularization
  - notation-robustness
---

# verifier-ce-xtune-qwen2.5-0.5b

> **Weights status: reproduction pending.** The recorded checkpoint was lost to
> a Colab runtime disconnect before upload; the model is fully reproducible from
> the committed recipe (`colab_augment.ipynb`, seed 9), and the notebook now
> uploads the checkpoint immediately after training. Until the re-run lands,
> the numbers below are the **recorded run's** (native logs
> `training_log_xtune_seed*.jsonl` + `results_augment.json`, committed). GPU
> runs are not guaranteed bit-exact across sessions, so on release this card
> will be updated with the released weights' own evaluation.

A **notation-robust** version of
[`verifier-ce-qwen2.5-0.5b`](https://huggingface.co/esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b):
the same AUTHORIZED / UNAUTHORIZED decision model, fine-tuned further with a
faithful xTune-style consistency-regularization recipe so that its verdicts no
longer depend on how resource identifiers are *notated*
(`cust:0/L1001` vs `cust:0:L1001` vs `cust|0|L1001`).

Companion model to the paper *"Learnable Authorization: A Verifier-as-Reward
Benchmark and Method for AI-Agent Authority."* Code, benchmark, and full
experiment log: https://github.com/esmaeil-abedi-dev/verifier-as-reward

## Why it exists

The base model transfers to real tool-call vocabulary (tau2-bench) at 90.8%
in its trained `family:namespace/leaf` notation but drops to 75.0% when the
same actions are re-rendered in an all-colon notation — the documented
spurious-format sensitivity of language models (Sclar et al., ICLR 2024), and
it fails *open* (false-authorize 50%). Naive notation augmentation makes this
worse (the augmentation-for-conventional-fine-tuning failure on fine-grained
tasks; Zheng et al., ACL 2021). This model applies their prescribed fix.

## Training (xTune stage 2; Zheng et al. ACL 2021, R-Drop NeurIPS 2021)

Fine-tuned from the released CE model, which also serves as the **frozen
stage-1 teacher** θ\*. Per training action:

- task cross-entropy on **both** the canonical and a re-notated view (labels
  identical — re-notation provably preserves the verifier's verdict);
- λ₁ · stop-gradient symmetric KL between the two views
  (`KL(sg(P)‖Q) + KL(sg(Q)‖P)`, R1);
- λ₂ · KL of both views to the frozen teacher (R2 — the collapse-preventer).

λ₁ = λ₂ = 5.0, lr 1e-5, 500 steps, batch 16, fp32. **Best-checkpoint
selection** on a synthetic transfer-validation set (`make_transfer_val.py`,
seed 303: real-mapping structure, synthetic vocabulary, mixed notations) —
training oscillates between notation-robust and brittle states, so the saved
checkpoint is the selection-best, never the last step. The released seed (9)
is the a-priori highest-selection seed (0.975).

## Evaluation (recorded run; every label from the verifier; 95% Wilson CIs in `results_augment.json`)

Held-out real tests, none seen in training:

| test | base model | **this model (seed 9)** | 3-seed mean |
|---|---|---|---|
| committed synthetic test (80) | 97.5 | 95.0 | 97.5 |
| tau2 real, slash notation (400) | 90.8 | **97.8** | 94.3 |
| tau2 real, colon notation (400) | 75.0 | **97.2** | 93.8 |
| Toucan balanced, mixed notations (2168) | 76.2 | **96.2** | 84.1 |
| Toucan authorized-only, mixed (1084) | 98.1 | 93.4 | 69.3 |

- The notation gap closes **upward** (colon 75.0 → 97.2) while the trained
  notation improves too (90.8 → 97.8).
- False-authorize on real tau2 falls from 18.5–50% to **4.5–5.5%**; the
  residual error direction flips from fail-open (over-authorization) to a
  small fail-closed over-refusal — the safer direction for authorization.
- Seed spread is real (see the 3-seed means; seeds 7/8 over-refuse on
  Toucan). The transfer-selection score rank-orders cross-source transfer
  (0.867/0.942/0.975 → Toucan-authorized 46.0/68.4/93.4), which is how seed 9
  is chosen without touching the real test sets.

## Intended use and limitations

- **Research artifact**, not a production authorization system; the
  deterministic verifier remains the ground truth.
- Robust to the four trained delimiter notations; **grant-structure
  brittleness remains** (a wildcard-action grant `perform '*'` still induces
  over-authorization — see the repo's REALTRACE_FINDINGS.md).
- The unauthorized half of the real tests is **constructed** (real calls
  redirected to real foreign ids); tau2/Toucan contain no naturally-occurring
  violations. See the real-vs-constructed accounting in the repo.
- Deterministic alternative: canonicalizing inputs to the trained notation at
  deployment recovers the base model's 90.8% with no retraining; the two
  mitigations are complementary.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("esmaeil-abedi-dev/verifier-ce-xtune-qwen2.5-0.5b")
model = AutoModelForCausalLM.from_pretrained("esmaeil-abedi-dev/verifier-ce-xtune-qwen2.5-0.5b")
```

Or evaluate with the repo's harness (no retraining):

```bash
PYTHONPATH=. python train_verifier_reward.py \
    --eval-checkpoint esmaeil-abedi-dev/verifier-ce-xtune-qwen2.5-0.5b \
    --test-file benchmark_test.jsonl
```
