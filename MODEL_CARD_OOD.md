---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
pipeline_tag: text-generation
tags:
  - authorization
  - ai-agent-safety
  - verifier-as-reward
  - out-of-distribution
---

# verifier-ood-qwen2.5-0.5b

The **out-of-distribution (domain hold-out) variant** of the verifier-CE
authorization model. Same method and base model as
[`verifier-ce-qwen2.5-0.5b`](https://huggingface.co/esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b),
but **deliberately trained on only 3 of the 5 domains** (email, payment,
repo) with **file and db held out**, so its transfer to unseen domains can
be measured honestly.

Companion to the paper *"Learnable Authorization: A Verifier-as-Reward
Benchmark and Method for AI-Agent Authority."* Code and full experiment log:
https://github.com/esmaeil-abedi-dev/verifier-as-reward

## Why this model exists

An out-of-distribution test is only valid if the test domains were never seen
in training. The main released model saw all 5 domains, so it cannot be used
to measure domain transfer. This model is trained without file/db precisely so
it can be evaluated **on** file/db as genuinely unseen domains.

## Evaluation (this checkpoint, seed 8)

| domains | accuracy | false-authorize | false-refuse |
|---|---|---|---|
| training (email/payment/repo) | 0.985 | 0.001 | 0.031 |
| **held-out (file/db) — OOD** | **0.970** | 0.008 | 0.059 |

Train→OOD gap ≈ 1.5 points (across 3 seeds: held-out 0.968 ± 0.001, train
0.983). The model transfers to entirely unseen action namespaces and resource
formats at near-parity — evidence that verifier-CE learns domain-invariant
authorization *structure* (delegation, attenuation, revocation, expiry,
budgets) rather than per-domain surface patterns. Residual error concentrates
in the two hardest structural classes (`chain_structure`, `scope_escalation`).

## Intended use and limitations

- **Research artifact** for the domain-transfer experiment — the main
  all-domains model is the one to use for general inference.
- Synthetic benchmark, single verifier as ground truth; transfer is shown
  across domain *vocabulary within the attenuated-delegation formalism*, not
  across different authorization formalisms.
- See the repository's `EXPERIMENT_LOG.md` for the full OOD result and the
  two-models rationale (§1b).

## Usage

```bash
PYTHONPATH=. python train_verifier_reward.py \
    --eval-checkpoint esmaeil-abedi-dev/verifier-ood-qwen2.5-0.5b \
    --test-file benchmark_test.jsonl
```
