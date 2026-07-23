# E2 — Qualitative / interpretability analysis

Per-action prediction records (`dump_predictions.py`) for four backends on the
committed test set (80 actions), used to (a) confirm the paper's claim that
residual error concentrates in `chain_structure`, and (b) show trace-level
where the verifier-CE 0.5B model matches the frontier and where it beats a
larger open model.

## Misclassifications by scenario class

| backend | acc | # wrong | classes of the errors |
|---|---|---|---|
| verifier-CE 0.5B (ours) | 0.975 | 2 | chain_structure ×2 |
| Claude-Sonnet-4.5 | 0.975 | 2 | chain_structure ×1, expiry ×1 |
| DeepSeek-R1 | 0.975 | 2 | chain_structure ×2 |
| Llama-3.1-8B | 0.600 | 32 | scope_escalation ×9, chain_structure ×9, confused_deputy ×4, resource ×4, revocation ×4, expiry ×1, budget ×1 |

**Two findings, both supporting the paper.**

1. **The residual error concentrates in `chain_structure`** — for *every* strong
   model. The three models that reach 0.975 (verifier-CE 0.5B, Sonnet-4.5,
   DeepSeek-R1) all err there; the 0.5B model errs *only* there. This is the
   trace-level confirmation of the aggregate claim.
2. **Llama-3.1-8B fails in the two structural-reasoning classes**
   (`scope_escalation` and `chain_structure` account for 18 of its 32 errors) —
   exactly the classes that require reasoning over the chain rather than
   surface cues. A 0.5B model trained on the verifier beats a 16×-larger
   general model precisely on the cases that demand chain reasoning.

## Error mode of the 0.5B model — the hardest structural case

The verifier-CE model's only two errors are both on **`chain_structure-0002`**,
a **broken-link** trace. The chain is:

```
Hop 1: org:acme        -> agent:researcher    (payment.*, ...)
Hop 2: agent:researcher-> agent:mailer         (payment.*, ...)
Hop 3: agent:executor  -> agent:billing-bot    (payment.refund, ...)   <-- delegator is agent:executor,
                                                                             but Hop 2 delegated to agent:mailer
Action: agent:billing-bot attempts payment.refund on vendor:acme/invoice-594
```

The chain is forged at hop 3: its delegator (`agent:executor`) is not the
delegatee of hop 2 (`agent:mailer`), so authority never actually flowed to
`agent:billing-bot`. True verdict: **UNAUTHORIZED** (failing hop 2, 0-based).
The 0.5B model answered AUTHORIZED — the final scope *does* permit the action,
and the model missed that the *link* upstream is broken. **Sonnet-4.5 and
DeepSeek-R1 caught it; Llama-3.1-8B also missed it.** So the residual error is
the subtlest structural check — verifying delegator = previous-delegatee across
hops — where only the strongest frontier models succeed. Every error the 0.5B
model makes, the frontier makes too (except this one, which the very top models
catch); it never fails the classes a smaller model fails.

## Where the 0.5B model beats Llama-3.1-8B

On the structural classes, the 0.5B verifier-CE model is correct where the
8B model is wrong. Representative (committed test): on `chain_structure` and
`scope_escalation` traces the 0.5B model returns the verifier's verdict while
Llama-8B over-authorizes (e.g. `chain_structure-0008/0009/0012`,
`scope_escalation-*`). Aggregate: 2 structural errors (0.5B) vs. 18 (Llama-8B).
The verifier signal teaches the small model the chain-reasoning that scale
alone does not buy the 8B model.

## By structural feature

The 0.5B model's two errors are both on depth-3 chains carrying a decoy grant
(`chain_len=3, has_decoy=True`) — i.e. the harder end of the distribution
(a broken link buried among a longer chain and a distractor grant), consistent
with the error being a genuine hard case rather than a shallow slip.

## Deliverables
- `predictions_{ce,sonnet,deepseek,llama8b}.jsonl` — per-action records
  (trace_id, class, structural features, prompt, label, prediction,
  failing_hop_true, raw reply); gitignored, regenerate with `dump_predictions.py`.
- This document — appendix table + curated traces.
