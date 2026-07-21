# E3 — Noise / robustness surrogate: findings

**Question.** Are the CE model's gains an artifact of clean templated prompts,
or does authorization judgment survive surface messiness?

**Method.** `make_noisy_test.py` perturbs the evaluation *prompt* only —
paraphrased instruction preamble, 1–3 interleaved irrelevant "tool telemetry"
log lines, benign typos in boilerplate, an irrelevant trailing note — while
keeping every load-bearing fact (principals, scopes, hop times, the action
line) verbatim. The verifier decides from structure, never the prompt, so the
label is invariant by construction; we re-run `label_action` on every noisy
trace and assert the label is unchanged.

**Design note (honest).** We deliberately do *not* reorder delegation lines:
in the attenuated-delegation model, hop order is load-bearing (each hop's
delegatee is the next hop's delegator), so reordering would change semantics,
not just surface form. The noise is therefore surface/prompt-level, which is
the correct robustness perturbation here.

## Result (committed test, 80 actions; released CE model, seed 9)

| condition | accuracy | 95% Wilson CI | false-authorize (headline) |
|---|---|---|---|
| clean | 97.5% | [91.3, 99.3] | 4.5% |
| noisy | 95.0% | [87.8, 98.0] | 2.3% |
| Δ | −2.5 pp | (CIs overlap) | −2.2 pp |

**Label invariance:** 0 of 80 actions changed label under noise (as expected by
construction; the guard confirms it).

## Reading

- The model **degrades gracefully**: a ~2.5-point accuracy drop under prompt
  noise, with the confidence intervals overlapping — no collapse. Accuracy
  stays well above the 80% lexical-heuristic floor and the false-authorize
  rate does not rise (it falls, i.e. the noise made it slightly more
  conservative, not more dangerous).
- This is evidence that the CE model's authorization judgment is not brittle to
  the exact clean template it was trained/evaluated on: irrelevant lines,
  reworded instructions, and typos in boilerplate do not break it.

## Limitation

This is a *surrogate* for real-log messiness, not real agent logs. It perturbs
the surface rendering of the same synthetic structures; it does not introduce
genuinely novel tool schemas or malformed structure. The external-validity
experiment (E5, real tau2 / AgentDojo traces) addresses the complementary
question of transfer to independently-authored traces. The 80-action test also
gives a wide CI; the clean/noisy comparison is a within-set paired contrast on
the same actions, which controls for that.
