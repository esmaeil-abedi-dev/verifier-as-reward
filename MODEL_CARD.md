---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
pipeline_tag: text-generation
tags:
  - authorization
  - ai-agent-safety
  - verifier-as-reward
  - rlvr
---

# verifier-ce-qwen2.5-0.5b

A **Qwen2.5-0.5B** model fine-tuned to decide **AUTHORIZED / UNAUTHORIZED**
for AI-agent actions over an object-capability delegation chain, using a
**deterministic authorization verifier as the sole training signal**
(verifier cross-entropy — the verdict as a target).

Companion model to the paper *"Learnable Authorization: A Verifier-as-Reward
Benchmark and Method for AI-Agent Authority."* Code, benchmark, and full
experiment log: https://github.com/esmaeil-abedi-dev/verifier-as-reward

## What it does

Given a natural-language description of a root authority, a chain of
delegations (each of which may only narrow the authority it received, with
issue/expiry/revocation times), and a pending action, the model answers
whether the action is authorized. The training target for every example is
the verdict of the deterministic verifier `verify(...)` — never a
hand-assigned or stored label.

## Training

- **Objective:** verifier cross-entropy, `-log π(verdict)` (see the paper /
  repo for why sampled REINFORCE and exact expected-reward policy gradient
  collapse or oscillate at this scale, and why the cross-entropy gradient's
  non-saturation is what converges).
- **Data:** synthetic execution traces across 5 domains and 9 scenario
  classes, generated and labeled by the verifier; leakage-guarded train /
  validation / test splits.
- **Recipe:** class-balanced reward weighting, natural-language prompts,
  full-parameter fine-tuning.

## Intended use and limitations

- **Research artifact**, not a production authorization system. The verifier
  itself is the ground truth; this model approximates it.
- Trained and evaluated on a **synthetic** benchmark from a single verifier;
  generalization to real agent logs or to authorization models outside the
  attenuated-delegation formalism is not established here.
- See the repository's `EXPERIMENT_LOG.md` for exact metrics (including the
  in-distribution vs. out-of-distribution domain-hold-out results) and their
  confidence intervals, and `DATASHEET.md` for the benchmark's limitations.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b")
model = AutoModelForCausalLM.from_pretrained("esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b")
```

Or evaluate it on any benchmark split with the repo's harness (no retraining):

```bash
PYTHONPATH=. python train_verifier_reward.py \
    --eval-checkpoint esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b \
    --test-file benchmark_test.jsonl
```
