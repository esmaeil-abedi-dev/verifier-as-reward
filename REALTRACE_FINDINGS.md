# E5 — Real-trace external validity: findings

**Question.** Does the verifier-trained model's authorization judgment transfer
to *independently-authored real agent tool-call traces* it never generated?

**Method.** `map_tau_to_chain.py` maps tau2-bench trajectories
(`Jarrodbarnes/tau2-sft-v4-dataset`, Apache-2.0; 219 trajectories: retail 112,
airline 49, telecom 58) into our `(root, chain, action)` schema, with **our
verifier assigning every label** (no hand-labeling). The support *system* holds
authority over all customers (`cust:*`) and delegates to the agent authority
over only the served customer (`cust:<tid>/*`, single hop). We extract the
agent's real tool calls that target a resource id (domain-agnostic: telecom
`L1001`, retail `#W2378156`, airline `EHGLP3`):
- **authorized** = the real in-scope call, resource `cust:<tid>/<id>` (200
  actions across the 3 domains — these are 100% real);
- **unauthorized** = the same real call redirected to a *different* customer's
  real id, `cust:<other>/<id>` — a confused deputy on a real call (the system
  could act on it; the agent's narrowed scope may not). Verifier-labeled.

400 balanced actions (200 authorized / 200 unauthorized). Every label
re-verifies under `label_action`; the mapper carries an offline fixture test
suite (7 tests).

**Scope / limits (state plainly in the paper):**
- **Single-hop** (tau2 is single-principal), so the attenuated-delegation
  *structure* is synthetic in the mapping; we test authorization judgment on
  real tool-call *semantics and vocabulary*, not real multi-hop delegation.
- The unauthorized cases are scope-violations **constructed** on real calls
  (real tool names, real args, real foreign ids — but the redirect is a
  perturbation, not a naturally-occurring attack).
- The redirect construction makes this a **resource-scope** task (in vs. out of
  the served customer's namespace), *not* a structural-reasoning one. The
  lexical-heuristic floor is therefore high here and is **reported alongside**
  the model — the model's contribution is transferring resource-scope judgment
  to real, unseen tool-call vocabulary, not solving the hard structural classes.

---

## A controlled sub-finding: the model is sensitive to resource *notation*

While building the mapping we ran, on the **same real data**, two resource
notations, and the contrast is itself a result:

| resource notation | example | CE-0.5B accuracy | false-authorize | heuristic |
|---|---|---|---|---|
| colon (`cust:0:L1001`, scope `cust:0:*`) — a notation the model never trained on | `cust:0:L1001` | **56.2%** [51.4, 61.0] | 87.5% | 100% |
| slash (`cust:0/L1001`, scope `cust:0/*`) — the trained `family:namespace/leaf` shape | `cust:0/L1001` | **[PENDING re-run]** | [PENDING] | 100% |

Under the colon notation the model **authorizes almost everything**
(false-authorize 87.5%) yet never wrongly refuses a real in-scope call
(false-refuse 0.0%) — it fails to recognize the out-of-scope resources because
their string shape differs from training, while the format-agnostic glob
heuristic is unaffected (100%). Re-mapping the identical real data into the
trained `family:namespace/leaf` shape isolates *vocabulary* transfer from
*notation* robustness.

**Takeaway (regardless of the pending number):** the model transfers across
resource **vocabulary** (real tool names and id formats — cf. the 0.97
zero-shot novel-domain result, which kept the trained notation) but is
**brittle to resource-string notation**. This is an honest, useful limitation:
a deployment would need to render resources in the notation the model was
trained on (or train across notations) — a one-line fix and a natural
robustness direction.

---

## Primary result — vocabulary transfer (corrected `family:namespace/leaf` notation)

**[PENDING the corrected Colab re-run — `colab_realtrace.ipynb` after the
mapper fix.]** Per-domain CE-0.5B accuracy / false-authorize / false-refuse
with 95% Wilson CIs, plus the heuristic floor:

| backend | telecom (n≈362) | airline (n≈20) | retail (n≈18) | all (n=400) |
|---|---|---|---|---|
| CE-0.5B (released) | — | — | — | — |
| lexical heuristic | 100% | 100% | 100% | 100% |

Interpretation to write once numbers land: the strongest sentence is the CE
model's **false-refuse rate on the 200 genuinely-real authorized calls**
(it correctly authorizes real, independently-authored tool calls it never
saw); the false-authorize rate on the redirects, reported next to the
heuristic floor, shows resource-scope judgment transferring to real vocabulary.

---

## Deliverables
- `map_tau_to_chain.py` (+ `test_map_tau_to_chain.py`, offline), `colab_realtrace.ipynb`
- `real_trace_{telecom,airline,retail,all}.jsonl` (gitignored; regenerate from seed 5)
- `results_realtrace.json` (per-domain + combined, +Wilson CIs)
- AgentDojo (native prompt-injection threat model) is cited as complementary
  future work, not run — a different formalism that would force verdict-vs-label
  reconciliation.
