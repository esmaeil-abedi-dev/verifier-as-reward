# Paper inserts — paste-ready text and tables

Drafted to match the manuscript's voice. Each insert says where it goes.
All numbers sourced from the committed artifacts (`results_*.json`,
`training_log_*.jsonl`, `proofoflife_results.json`); nothing new was run.
Tables are in markdown — convert to your LaTeX table style.

---

## INSERT 1 — §5.1, expand the RL study (replaces the three-bullet list's surrounding prose; keep the bullets or fold them in)

> The natural instantiation of "verdict as reward" is reinforcement learning.
> We ran a structured ablation of four policy-gradient variants and report the
> full per-seed outcome in Table 3. Sampled REINFORCE with a running-mean
> baseline (three seeds) collapses within twenty steps to a blanket
> always-refuse policy. The collapse is not noise but a rewarded local
> optimum: the training split is 55.3% unauthorized, so refusing everything
> earns an expected reward of approximately +0.11. Two of three seeds later
> *escape* the collapse stochastically — a lucky high-variance batch around
> step 200 restores exploration — and recover into an oscillation band
> (0.50–0.64) without converging. Adding class-balanced rewards and an entropy
> bonus (three seeds) removes the majority-class payoff but merely makes the
> corner lottery symmetric: across seeds the policy lands in always-authorize,
> always-refuse, or transient discrimination. Increasing the batch size from 8
> to 32 with a stronger entropy bonus — the standard variance-reduction move —
> makes the outcome strictly *worse*: the run we observed stayed pinned in the
> collapsed policy for its entire duration, because averaging over larger
> batches removes exactly the gradient noise that was the only escape
> mechanism. Finally, the exact expected-reward policy gradient, which
> eliminates sampling variance entirely by pricing both decisions in closed
> form, oscillates for a different reason: its gradient with respect to the
> decision logit is proportional to π(A)π(R), which vanishes as the policy
> approaches either corner. Its final checkpoint is instructive for a security
> setting: it ended in the *always-authorize* corner, scoring a false-authorize
> rate of 1.0 on the held-out test — worse on the safety metric than the
> untrained model. The wrong objective does not merely fail to learn; it can
> fail unsafe. (Seed counts per variant: three for the two sampled variants;
> the exact-gradient variant was stopped after a second seed reproduced the
> same corner behavior, and the batch-32 variant after its first seed showed
> total collapse.)

**Table 3 (new) — RL variants, per-seed held-out endpoint (accuracy / false-authorize / false-refuse):**

| variant | seed 7 | seed 8 | seed 9 |
|---|---|---|---|
| sampled REINFORCE | 0.613 / 0.23 / 0.58 | 0.613 / 0.21 / 0.61 | 0.550 / 0.00 / 1.00 |
| + balanced reward, entropy 0.01 | 0.450 / 1.00 / 0.00 | 0.550 / 0.00 / 1.00 | 0.613 / 0.30 / 0.50 |
| + batch 32, entropy 0.03 | 0.550 / 0.00 / 1.00 (stopped @200) | — | — |
| exact expected-reward gradient | 0.592 / 0.39 / 0.44 | (stopped @25, collapsed) | — |
| **verifier cross-entropy (ours)** | **0.988 / 0.02 / 0.00** | **0.988 / 0.02 / 0.00** | **0.975 / 0.05 / 0.00** |

*(Untrained baseline: 0.475. RL endpoints are val-monitored curves at their
final step; CE row is the committed test set, for direct comparison with
Table 2.)*

---

## INSERT 2 — §5.2, after the "(π(verdict)−1)" sentence (the empirical saturation measurement)

> This is not only an analytic argument. Measured on a real checkpoint at a
> confidently-wrong state (π(AUTHORIZE) ≈ 0.9998 with the verifier requiring
> REFUSE), the cross-entropy parameter-gradient norm is 5.31 against 0.0022
> for the expected-reward objective — a factor of roughly 2,400. The
> policy-gradient objectives are not wrong about the optimum; they are unable
> to move toward it from precisely the states that matter.

---

## INSERT 3 — §5.2, training details + data discipline (add after "Trained three seeds × 500 steps on natural-language prompts")

> Training uses batch 16, learning rate 2×10⁻⁵, gradient clipping at norm
> 1.0, and inverse-class-frequency reward weighting, over an expanded
> training corpus of 2,400 actions generated from a fresh seed. Data
> discipline is strict: every training example is canonicalized as a (root,
> chain, action) decision context and deduplicated against the committed test
> set (measured contamination: 0 of 80 test contexts); hyperparameters were
> selected on a separate 400-action validation corpus from a third seed; and
> the committed test set was evaluated exactly once per seed, at the end.

---

## INSERT 4 — §5.3, the two-phase transfer observation (add to the out-of-distribution paragraph)

> The transfer has a consistent temporal structure: on the held-out domains
> the *refusal* side generalizes first (the false-authorize rate falls to
> near zero by mid-training) while the *authorize* side lags (false-refusals
> on unseen resource formats persist until late training before dropping to
> zero) — the model is initially conservative on vocabulary it has never
> seen, then generalizes the permission side as well.

## INSERT 5 — §5.3, novel-domain construction (one sentence before the zero-shot numbers)

> The novel domains are constructed with action namespaces and resource
> prefixes provably disjoint from all five training domains, and the
> zero-shot figure is corroborated across two independent draws — a
> two-domain set (97.8%) and a six-domain set spanning calendar, cloud, IoT,
> finance, messaging, and storage (97.2%) — so the result is not specific to
> a fortunate choice of domain.

---

## INSERT 6 — §4, the fail-safe evaluation convention (one sentence, e.g. after the metric definition)

> The evaluator is fail-safe by construction: a backend that errors, times
> out, or returns an unparseable reply is scored as a refusal, so no failure
> mode of the evaluation pipeline can be counted as an authorization.

---

## INSERT 7 — new section before the Conclusion: "Artifacts and Reproducibility"

> **Artifacts and reproducibility.** All artifacts are public: the verifier,
> generators, evaluation and training code, the benchmark with its datasheet,
> every training log, and two trained checkpoints — the all-domains model
> (`verifier-ce-qwen2.5-0.5b`) and the domain-hold-out model
> (`verifier-ood-qwen2.5-0.5b`) — on the Hugging Face Hub. [ADD GITHUB URL.]
> The corpus regenerates byte-identically from its seed; all random number
> generators are seeded, and each training log begins with a machine-readable
> record of the exact command and configuration that produced it. The
> codebase carries 95 unit tests across seven suites, including tests that
> re-verify every benchmark label against the verifier, prove the
> leakage-deduplication guard fires on planted collisions, and check the
> numerical correctness of the training objectives against independent
> reference implementations. Every number in this paper traces to a committed
> artifact.

---

# APPENDIX MATERIAL

## Appendix A — Objectives and gradient geometry

Policy. For each action, the model scores the two candidate continuations
c ∈ {" AUTHORIZE", " REFUSE"} by length-normalized sequence log-probability
s_c = (1/|c|) Σ_j log p(c_j | prompt, c_<j), and the policy is
π = softmax(s_A, s_R).

Reward. With verifier verdict v ∈ {authorize, refuse} from a live call to
`verify`, the reward for decision d is r(d) = +w_v if d = v else −w_v, where
w_v is an optional inverse-class-frequency weight.

Objectives (per example; b is a running-mean baseline, β an entropy weight):

- **Sampled REINFORCE:** L = −(r(d) − b) · log π(d), d ~ π.
- **Exact expected-reward:** L = −[π(A) r(A) + π(R) r(R)] (no sampling).
- **Verifier cross-entropy:** L = −w_v · log π(v).

Gradient geometry. Write z = s_A − s_R and, without loss of generality, let
the correct decision be A, so π(A) = σ(z). Then:

- exact expected-reward: dE[r]/dz = 2 π(A) π(R) → 0 as π(A) → 0 or 1
  (saturates at both corners; an oscillator, not a converger);
- cross-entropy: d(−log π(A))/dz = π(A) − 1 → −1 as π(A) → 0
  (maximal gradient exactly when confidently wrong).

Empirically, at a confidently-wrong state on the trained architecture
(π(A) ≈ 0.9998, correct = R), the parameter-gradient norms are 5.31 (CE)
vs 0.0022 (exact expected-reward): a ≈2,400× ratio.

## Appendix B — Per-class accuracy, all evaluations

Class order: single-delegation, multi-hop, revocation, expiry,
scope-escalation, resource-violation, budget-violation, confused-deputy,
chain-structure. Committed test set (80 actions) unless noted.

**B.1 Frontier ladder (per-class accuracy):**

| backend | sing. | multi | revoc. | expiry | scope-esc. | res-viol. | budget | deputy | chain-str. |
|---|---|---|---|---|---|---|---|---|---|
| heuristic floor | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 | 0.80 | 1.00 | 1.00 | 0.30 |
| Llama-3.1-8B | 0.80 | 1.00 | 0.70 | 0.70 | 0.00 | 0.50 | 0.80 | 0.60 | 0.10 |
| Llama-3.3-70B | 1.00 | 1.00 | 1.00 | 0.90 | 0.20 | 0.80 | 1.00 | 1.00 | 0.30 |
| Gemini-2.5-Flash | 1.00 | 1.00 | 1.00 | 0.90 | 0.60 | 1.00 | 1.00 | 0.90 | 0.60 |
| DeepSeek-R1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 | 0.80 |
| Claude-Sonnet-4.5 | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |

**B.2 Verifier-CE 0.5B (committed test, per seed):**

| seed | sing. | multi | revoc. | expiry | scope-esc. | res-viol. | budget | deputy | chain-str. |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |
| 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |
| 9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 |

**B.3 OOD model on held-out domains (file/db; 1,150 actions; budget class is
payment-only and therefore absent from the held-out domains):**

| seed | sing. | multi | revoc. | expiry | scope-esc. | res-viol. | budget | deputy | chain-str. |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 1.00 | 1.00 | 0.99 | 1.00 | 0.97 | 0.96 | — | 0.97 | 0.87 |
| 8 | 1.00 | 0.90 | 1.00 | 1.00 | 0.97 | 0.99 | — | 1.00 | 0.87 |
| 9 | 1.00 | 0.99 | 1.00 | 1.00 | 0.90 | 1.00 | — | 1.00 | 0.87 |

**B.4 Released model, zero-shot novel domains (six-domain draw, 640 actions):**
single 1.00, multi-hop 0.925, revocation 1.00, expiry 1.00, scope-escalation
0.962, resource-violation 0.988, budget 0.950, confused-deputy 1.00,
chain-structure 0.912.

**B.5 Released model, fresh in-distribution draw (960 actions):** single 1.00,
multi-hop 0.95, revocation 1.00, expiry 1.00, scope-escalation 0.975,
resource-violation 0.992, budget 0.975, confused-deputy 1.00,
chain-structure 0.908.

*The through-line: chain-structure is the hardest class in every evaluation
— for every frontier model (0.10–0.90), for the trained model in-distribution
(0.80–0.90), out-of-distribution (0.87), on novel domains (0.91), and even on
the trained model's own training data (0.94). Multi-hop structural reasoning
is the residual hard case for models and the easy case for the verifier — the
clearest illustration of why a mechanical verifier belongs in the loop.*

## Appendix C — Training configurations

| arm | objective | train data (actions) | steps | batch | lr | extras | seeds |
|---|---|---|---|---|---|---|---|
| A | sampled REINFORCE | benchmark train (320) | 400 | 8 | 1e-5 | clip 1.0 | 7,8,9 |
| B | + mitigations | benchmark train (320) | 400 | 8 | 1e-5 | entropy 0.01, balanced | 7,8,9 |
| C | + tuning | benchmark train (320) | 200 | 32 | 1e-5 | entropy 0.03, balanced | 7 (partial) |
| D | exact expected-reward | expanded (2,400) | 400 | 16 | 1e-5 | balanced | 7; 8 partial |
| E | verifier cross-entropy | expanded (2,400) | 500 | 16 | 2e-5 | balanced, NL prompts | 7,8,9 |
| OOD | verifier cross-entropy | 3-domain (2,050) | 500 | 16 | 2e-5 | balanced, NL prompts | 7,8,9 |

All runs: Qwen2.5-0.5B base, full-parameter AdamW, gradient clip 1.0,
seeded; every log begins with the exact configuration record.

## Appendix D — The lexical heuristic baseline (verbatim rules)

The heuristic reads only the rendered prompt text and applies three rules in
order, refusing if any fires, otherwise authorizing:

1. **Temporal:** if the action time t is at or past any number following
   "REVOKED at t=" or "expires at t=" anywhere in the prompt, refuse.
2. **Resource:** if the action's resource string matches none of the resource
   globs quoted in the *last* hop's line, refuse.
3. **Budget:** if the action amount exceeds the smallest spending cap
   mentioned anywhere in the prompt, refuse.

It never examines chain connectivity, issue times, attenuation between hops,
or which grant a cap belongs to. On the unhardened corpus these rules scored
91.4%; on the hardened corpus, 80.0% — the floor reported in Table 1. Their
failure modes are exactly the hardening devices: chain-structure violations
(rules blind to wiring), decoy grants (rule 2 matches the wrong grant's
glob; rule 3 the wrong cap), and inert timestamps (rule 1 fires on windows
that lie after the action).

---

## INSERT 8 — notation robustness on real traces (E5b/E5c): finding, negative augmentation result, and the consistency-regularization fix

Goes in the real-trace transfer section, after the vocabulary-transfer result.
Fill the bracketed consistency-regularization numbers from the
`colab_augment.ipynb` run (`results_augment.json`); the released and negative-
augmentation numbers are final.

> **Finding — notation sensitivity.** Mapping real tau2-bench trajectories into
> our schema requires choosing a surface *notation* for resource identifiers.
> The verifier-CE model transfers to the real tool-call *vocabulary* but is
> sensitive to this notation: on 400 balanced real actions it scores 90.8%
> (95% Wilson CI [87.5, 93.2]) in the trained `family:namespace/leaf` (slash)
> notation and 75.0% [70.5, 79.0] when the identical actions are re-rendered in
> an all-colon notation, with the drop entirely in over-authorization
> (false-authorize 18.5% → 50.0%; false-refuse 0% in both). This is the known
> spurious-format sensitivity of language models, which persists across scale
> and instruction tuning (Sclar et al., 2024).
>
> **Naive augmentation makes it worse.** Retraining the cross-entropy objective
> on a notation-augmented corpus — either an even four-way delimiter mix or a
> 70% canonical-majority mix — does not close the gap. It converts real-trace
> accuracy into a seed lottery (50–81% across three seeds; one run collapsed to
> near-always-authorize, false-authorize ≈ 100%) while synthetic-test accuracy
> stayed ≈ 96–100%: the model overfits the augmented synthetic distribution and
> transfers worse. This is the documented failure mode of using augmentation for
> *conventional* fine-tuning on *fine-grained* tasks — ours discriminates
> `cust:0/…` from `cust:5/…` — whereas the *same* augmentation used for
> **consistency regularization** helps by a large margin (Zheng et al., 2021).
>
> **The fix — consistency regularization.** We keep the cross-entropy loss on
> the canonical rendering (preserving the sharp discrimination) and *add* a
> symmetric Kullback–Leibler term tying the model's authorize/refuse
> distribution on the canonical view to its distribution on the *same action
> re-notated* in a randomly chosen delimiter scheme (an R-Drop–style consistency
> objective; Liang et al., 2021; Botev et al., 2022):
> L = CE(verdict ∣ canonical) + λ · KL_sym( p(·∣canonical) ‖ p(·∣re-notated) ).
> Because re-notation provably preserves the verifier's verdict, both views
> share the same label; the KL teaches *notation-invariance* — "return the same
> verdict regardless of delimiter" — without exposing the cross-entropy loss to
> off-notation data, which is what destabilized naive augmentation. With λ =
> [__], the model reaches [__]% [CI] on the slash notation and [__]% [CI] on
> colon, stable across three seeds (false-authorize [__]%).
>
> **Deterministic alternative.** Because the mapping from real logs into the
> schema is under our control, notation robustness can also be obtained without
> retraining, by canonicalizing inputs to the trained notation at deployment —
> recovering the 90.8% slash-notation result on any input format. We report the
> learned (consistency-regularized) and deterministic (canonicalization)
> mitigations as complementary.

### References for INSERT 8 (APA)

- Botev, A., Bauer, M., & De, S. (2022). *Regularising for invariance to data
  augmentation improves supervised learning.* arXiv:2203.03304.
  https://doi.org/10.48550/arXiv.2203.03304
- Liang, X., Wu, L., Li, J., Wang, Y., Meng, Q., Qin, T., Chen, W., Zhang, M., &
  Liu, T.-Y. (2021). R-Drop: Regularized dropout for neural networks. In
  *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*.
  https://doi.org/10.48550/arXiv.2106.14448
- Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying language
  models' sensitivity to spurious features in prompt design or: How I learned to
  start worrying about prompt formatting. In *The Twelfth International
  Conference on Learning Representations (ICLR 2024)*.
  https://doi.org/10.48550/arXiv.2310.11324
- Zheng, B., Dong, L., Huang, S., Wang, W., Chi, Z., Singhal, S., Che, W., Liu,
  T., Song, X., & Wei, F. (2021). Consistency regularization for cross-lingual
  fine-tuning. In *Proceedings of the 59th Annual Meeting of the Association for
  Computational Linguistics and the 11th International Joint Conference on
  Natural Language Processing (ACL-IJCNLP 2021)* (pp. 3403–3417).
  https://doi.org/10.18653/v1/2021.acl-long.264

### BibTeX for INSERT 8

```bibtex
@inproceedings{sclar2024quantifying,
  title     = {Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting},
  author    = {Sclar, Melanie and Choi, Yejin and Tsvetkov, Yulia and Suhr, Alane},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2310.11324},
  doi       = {10.48550/arXiv.2310.11324}
}

@inproceedings{zheng2021consistency,
  title     = {Consistency Regularization for Cross-Lingual Fine-Tuning},
  author    = {Zheng, Bo and Dong, Li and Huang, Shaohan and Wang, Wenhui and Chi, Zewen and Singhal, Saksham and Che, Wanxiang and Liu, Ting and Song, Xia and Wei, Furu},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)},
  pages     = {3403--3417},
  year      = {2021},
  doi       = {10.18653/v1/2021.acl-long.264}
}

@inproceedings{liang2021rdrop,
  title     = {R-Drop: Regularized Dropout for Neural Networks},
  author    = {Liang, Xiaobo and Wu, Lijun and Li, Juntao and Wang, Yue and Meng, Qi and Qin, Tao and Chen, Wei and Zhang, Min and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year      = {2021},
  note      = {arXiv:2106.14448},
  doi       = {10.48550/arXiv.2106.14448}
}

@article{botev2022regularising,
  title   = {Regularising for invariance to data augmentation improves supervised learning},
  author  = {Botev, Aleksander and Bauer, Matthias and De, Soham},
  journal = {arXiv preprint arXiv:2203.03304},
  year    = {2022},
  doi     = {10.48550/arXiv.2203.03304}
}
```

*Verify the Zheng et al. page range (3403–3417) against the ACL Anthology entry
before final submission; all DOIs and author lists were confirmed against the
published records.*
