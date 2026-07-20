# Related Work and Next-Step Directions

Literature grounding for the "RL + something else" question — why our pure
verifier-RL runs (Arms A–D in `EXPERIMENT_LOG.md`) failed the way they did,
what the field's established fixes are, and which we should adopt. Each source
is summarized with the specific detail that bears on our setup, not a
one-line gloss. All sources were fetched and read (July 2026); URLs at the end.

Our observed failures, for reference (see `EXPERIMENT_LOG.md` §4):
- Sampled REINFORCE (Arms A/B/C) collapses to a blanket decision — an
  **exploration/initialization** failure from a near-random 0.5B base policy.
- Exact policy gradient (Arm D) removes sampling variance but its gradient
  ∝ π(A)π(R) **saturates at the corners**, so it oscillates and its checkpoint
  ended at false-authorize 1.0.
- Verifier cross-entropy (Arm E, `--ce-loss`) has a non-saturating gradient
  and is the pending headline run.

---

## 1. RLVR genuinely teaches reasoning — it is not just re-sampling base skills

**Wen, Liu, Zheng, et al. — "Reinforcement Learning with Verifiable Rewards
Implicitly Incentivizes Correct Reasoning in Base LLMs." arXiv:2506.14245v2.**

- **Claim relevant to us:** RLVR (RL with a verifiable/deterministic reward —
  exactly our verifier setup) *fundamentally enhances* reasoning rather than
  merely improving sampling efficiency. It disputes the "base model already
  contains all the reasoning paths, RL just up-weights them" view.
- **Method detail:** introduces a **CoT-Pass@K** metric that scores both the
  final answer *and* the intermediate reasoning steps, not just the answer.
  On AIME 2025, the reasoning-quality gap between DAPO-Qwen-32B and its base
  Qwen2.5-32B **persists across all K up to 1024** — i.e. RL added reasoning
  the base model could not reach by sampling alone.
- **The load-bearing sentence for our thesis:** supervised fine-tuning on
  RLVR-*generated* CoT data "nearly replicates the performance of a post-RLVR
  model" — reasoning patterns from RLVR are **genuinely learned**, not dormant.
- **Why it matters here:** (a) it justifies the whole premise that a verifier
  reward can teach a policy something real; (b) the SFT-replicates-RLVR result
  is direct support for our Arm E move — using the verifier as a *supervised
  target* (CE) can capture most of what RL would, which is why CE is a
  legitimate and strong instantiation of "verifier as reward," not a cop-out.

## 2. Small models need a warm start before RL — pure RL from base is unstable

**DeepSeek-R1 (as explained in the NormalUhr HuggingFace write-up).**

- **GRPO, precisely:** Group Relative Policy Optimization drops the critic/value
  network and computes each sample's advantage by **group-relative
  normalization**: `A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)` over a group of
  G sampled responses to the same prompt. This is the variance-reduction our
  scalar-baseline REINFORCE lacked (Arms A–C).
- **Cold-start necessity:** DeepSeek-R1-Zero (pure RL from base, *no* SFT) had
  strong reasoning but "output was often in tangles — mixing languages, lacking
  user-friendly structure." The fix was to "inject a small supervised
  'cold-start' dataset (thousands of curated chain-of-thought samples)" *before*
  RL.
- **Full pipeline:** cold-start SFT → reasoning-focused RL (with extra
  language-consistency rewards) → rejection sampling → a second RL stage.
- **Why it matters here:** this is the canonical "RL + something else." Our
  Arms A–D are the R1-Zero analogue (RL from a cold base) and failed for the
  analogous reason — no foundational task competence for RL to refine. The
  recommended structure is **warm-start then RL**, and in our setting the
  verifier supplies *both* stages (CE labels, then RL reward).

## 3. Cold-start SFT + RL beats either alone — with concrete gains

**Lai, Li, Zheng, Wang, et al. — "Advancing Multimodal Reasoning via
Reinforcement Learning with Cold Start." arXiv:2505.22334.**

- **Result:** the combined SFT-then-RL recipe "consistently outperforms both
  SFT-only and RL-only methods across challenging reasoning benchmarks."
- **Recipe:** Stage 1 = SFT on structured CoT (distilled by rejection sampling
  from a larger 32B model); Stage 2 = GRPO refinement.
- **Numbers (Qwen2.5-VL-7B):** 66.3% → 73.4% on MathVista, 62.9% → 70.4% on
  We-Math over baseline; the **3B** variant became competitive with several 7B
  models — evidence that the recipe makes *small* models punch above their size.
- **Why it matters here:** direct quantitative precedent that warm-start + RL
  lifts small models, and that RL adds real points *on top of* SFT (not just
  redundant with it) — the shape of result we want for a positive RL story.

## 4. But the warm-start must be good — the "valley" caution for tiny models

**Luo, Li, Huang, Lu — "Through the Valley: Path to Effective Long CoT Training
for Small Language Models." arXiv:2506.07712v2.**

- **The valley phenomenon:** small models trained SFT-on-CoT → RL suffer a
  **performance dip at intermediate stages** before (sometimes) recovering.
- **SFT-quality warning:** "SFT quality" is a fundamental constraint — poor or
  misaligned SFT data undermines the subsequent RL stage (echoed by the search
  result that low-quality CoT supervision can *distort* a strong base and make
  a *worse* RL init than the base itself).
- **Size threshold:** the study explicitly examines **Qwen2.5-0.5B-Instruct**
  (our exact scale) and finds models below a capacity threshold struggle
  particularly with long-CoT sequences.
- **Why it matters here — two concrete takeaways:**
  1. Our warm-start data quality is *not* a risk: our CE labels are the exact
     verifier verdicts (perfect, deterministic), not distilled noisy CoT — so
     we avoid the paper's main failure cause.
  2. It is a real caution about 0.5B specifically. If CE alone plateaus below
     target, the valley result argues for either (a) a larger model, or (b)
     shorter/structured reasoning rather than long free-form CoT.

## 5. Verifiable-reward RL in practice — reward design and reward hacking

**AWS — "Overcoming reward signal challenges: Verifiable rewards-based RL with
GRPO on SageMaker AI."**

- **Core challenge:** imprecise reward functions enable **reward hacking** —
  the policy maximizes the score without the intended behavior ("hidden biases,
  unintended incentives, ambiguous success criteria").
- **Recommended design:** *dual* rule-based rewards — a **format reward**
  (structure, e.g. 0.5 for a required answer pattern) plus a **correctness
  reward** (e.g. 1.0 for the right value, with a tolerance for floats). Works
  best when outputs are objectively verifiable — math, code, symbolic tasks.
- **Why it matters here:** our verifier is already a perfect, non-hackable
  correctness reward, so we sidestep the paper's central worry. The relevant
  transferable idea is the **dual-reward decomposition**: if we add CoT, a
  small format reward (decision emitted in the parseable slot) plus the
  verifier correctness reward is the standard, safe structure — and it prevents
  the model from "reasoning" its way out of ever committing to a decision.

## 6. Curated list (for the paper's broader RLVR citations)

**opendilab/awesome-RLVR** — continually-updated curated list of RL-with-
verifiable-rewards papers; useful as the survey anchor when we write the
related-work paragraph and need breadth beyond the six sources above.

---

## 7. Synthesis — what "RL + something else" should be, mapped to our failures

| our observed failure | literature's diagnosis | the fix |
|---|---|---|
| REINFORCE collapse (Arms A/B/C) | cold-start RL on a near-random base has no signal to refine (§2 R1-Zero) | **warm-start (CE) then RL** (§2, §3) |
| exact-PG saturation/oscillation (Arm D) | expected-reward gradient dies at the corners | non-saturating loss = CE (Arm E); then RL from that point |
| high variance of the scalar-baseline REINFORCE | no group-relative advantage | **GRPO** (§2) — group-normalized advantage, no critic |
| hard classes unsolved one-shot (`chain_structure`, `scope_escalation`) | multi-step problems need working memory | **chain-of-thought before the decision** (§1, §2), with a dual format+correctness reward (§5) |
| risk that a warm-start hurts | low-quality SFT distorts the base (§4 valley) | our CE labels are perfect verifier verdicts — low risk; watch the 0.5B valley (§4) |

**Recommended sequence for our next runs (keeps RL in the thesis honestly):**
1. **CE warm-start to convergence** (Arm E, `--ce-loss --prompt-style nl`) —
   already implemented; targets the 75–85% band on its own.
2. **RL-refine from the CE checkpoint** — GRPO-style group-relative advantage,
   with a **KL anchor to the CE policy** so RL refines calibration / lowers
   false-authorize instead of wandering back to a corner (the Arm-D failure).
   The same verifier supplies this reward — no new signal source.
3. **If the hard classes stay stuck:** add **chain-of-thought before the
   decision** with a dual format + verifier-correctness reward (§5), the
   R1-style recipe (§2) — this is where the `chain_structure` /
   `scope_escalation` points live (even llama-70B scored 0.20–0.30 there).

The honest framing for the paper: pure verifier-RL fails at 0.5B by
optimization pathology; the verifier's power is realized first as a
**supervised target** (CE, ≈ the SFT-replicates-RLVR result of §1), and RL
then **refines** that policy (§2, §3) — the standard, evidence-backed
warm-start-then-RL pipeline, with the same deterministic verifier driving
both stages.

---

## Sources

1. Wen, X., Liu, Z., Zheng, S., et al. *Reinforcement Learning with Verifiable
   Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs.*
   arXiv:2506.14245v2. https://arxiv.org/html/2506.14245v2
2. *DeepSeek-R1 explained: how RL masters complex reasoning* (GRPO, cold-start,
   multi-stage pipeline). https://huggingface.co/blog/NormalUhr/deepseek-r1-explained
3. Lai, W., Li, Y., Zheng, K., et al. *Advancing Multimodal Reasoning via
   Reinforcement Learning with Cold Start.* arXiv:2505.22334.
   https://github.com/waltonfuture/RL-with-Cold-Start
4. Luo, R., Li, J., Huang, C., Lu, W. *Through the Valley: Path to Effective
   Long CoT Training for Small Language Models.* arXiv:2506.07712v2.
   https://arxiv.org/pdf/2506.07712
5. AWS Machine Learning Blog. *Overcoming reward signal challenges: Verifiable
   rewards-based reinforcement learning with GRPO on SageMaker AI.*
   https://aws.amazon.com/blogs/machine-learning/overcoming-reward-signal-challenges-verifiable-rewards-based-reinforcement-learning-with-grpo-on-sagemaker-ai/
6. opendilab. *awesome-RLVR* (curated list).
   https://github.com/opendilab/awesome-RLVR
7. GRPO Training Pipeline: SFT to RL for Better Reasoning.
   https://langcopilot.com/posts/2025-09-05-grpo-training-pipeline-sft-rl-better

*Note: items 1, 3, 4 are peer-reviewable arXiv preprints suitable for direct
citation; items 2, 5, 7 are technical blog posts (use for background/rationale,
prefer the underlying papers — the DeepSeek-R1 and GRPO source papers — for
formal citations); item 6 is a link collection.*
