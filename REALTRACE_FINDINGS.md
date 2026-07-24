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

## What is real vs. constructed (read this before claiming "real data")

A reviewer will ask exactly how much of this is real. Component-by-component:

| element of each action | provenance |
|---|---|
| tool name (`suspend_line`, `get_reservation_details`) | **REAL** — verbatim from tau2 |
| tool arguments / resource-id values (`L1001`, `#W2378156`, `EHGLP3`) | **REAL** — verbatim from tau2 |
| numeric amount (when present) | **REAL** — verbatim from tau2 |
| **authorized** action = (this agent, this tool, its served customer's id) | **REAL** — the call the agent actually made |
| **unauthorized** action = (this tool, a *different* customer's real id) | **CONSTRUCTED** — real ingredients, synthetic pairing (the redirect never occurred in the log) |
| root / delegation / scope grants | **CONSTRUCTED** — our formalism, imposed on single-principal traces (tau2 has no delegation) |
| every label | **our VERIFIER** — applied to the constructed scope structure |

**The precise, defensible claims** (use these words; do not round up):
- ✅ "The model correctly authorizes **real, independently-authored in-scope
  tool calls** it never saw" — the 200 authorized actions are real calls;
  0% false-refuse. This is the strongest and cleanest sentence.
- ✅ "Its authorization judgment **transfers to real tool-call vocabulary**
  (tool names and resource-id formats from an independent benchmark)."
- ⚠️ NOT "evaluated on real authorization logs" — there is no such corpus;
  tau2 is good-behavior data with no naturally-occurring violations, so the
  **unauthorized half is constructed** (real calls redirected to real foreign
  ids). Say "constructed scope-violations on real calls."
- ⚠️ NOT "90.8% on real data" unqualified — that number mixes the real
  (authorized) and constructed (unauthorized) halves. Report the two halves
  separately: **0% false-refuse on real in-scope calls** (the real-data
  result) and **18.5% false-authorize on constructed redirects** (the
  constructed-violation result).

Why the unauthorized half must be constructed: tau2 is supervised
good-behavior data — the agents stay in scope — so there are no natural
out-of-scope actions to label 0. A balanced test therefore requires
constructing violations; we do so from real calls and real foreign ids, and
the verifier (not us) labels them. This is disclosed, not hidden.

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

## A controlled sub-finding: the model is sensitive to the *representation*, not just the vocabulary

Getting a faithful mapping surfaced a genuine, reportable property. On the
**same real data** we varied how the authorization structure is *rendered*
into our schema (the verifier labels are identical and correct in every
version; only the surface representation the model reads changes):

| mapping representation | CE-0.5B accuracy | false-authorize | false-refuse | heuristic |
|---|---|---|---|---|
| colon resource `cust:0:L1001` (notation the model never trained on) | 56.2% | 87.5% | 0.0% | 100% |
| slash `cust:0/L1001` but wildcard-action grant `perform '*'` | 55.2% | 89.5% | 0.0% | 100% |
| slash + **specific-action** grants (the trained representation) | **90.8%** [87.5, 93.2] | 18.5% | 0.0% | 100% |

Two representation shifts each break the model while leaving the verifier and
the format-agnostic heuristic unaffected: (i) resource **notation** (colon vs.
the trained `family:namespace/leaf` slash), and (ii) grant **structure** (a
wildcard-action grant `perform '*'` reads as "may do anything" → the model
over-authorizes; the trained grants always name a concrete action). In both
broken versions the model still *authorizes every real in-scope call* (0%
false-refuse) — it fails only to *refuse* out-of-scope access, i.e. it
over-authorizes when the representation is off-distribution.

When the mapping uses the trained representation (slash resources +
specific-action grants) with the **real, unseen vocabulary** (real tool names
`suspend_line`/`get_reservation_details`, real id formats `L1001`/`#W2378156`/
`EHGLP3`), the model transfers: a 10-action local diagnostic scored 10/10,
correctly authorizing in-scope real calls and refusing out-of-scope redirects.

**Takeaway (honest and useful):** the model transfers authorization judgment
to real tool-call *vocabulary*, but is **sensitive to the representation** of
resources and grants — it must be fed the notation and grant structure it was
trained on. That is a real robustness limitation with a clear mitigation
(normalize the representation at deployment, or train across representations),
and it is more informative to report than to hide.

---

## Primary result — vocabulary transfer (trained representation, real tau2 tool calls)

Released CE-0.5B vs the lexical-heuristic floor on the mapped real traces
(accuracy [95% Wilson CI], false-authorize, false-refuse):

| domain | n | CE-0.5B accuracy | CE false-auth | CE false-refuse | heuristic |
|---|---|---|---|---|---|
| telecom | 362 | 91.4% [88.1, 93.9] | 17.1% | 0.0% | 100% |
| airline | 20 | 90.0% [69.9, 97.2] | 20.0% | 0.0% | 100% |
| retail | 18 | 77.8% [54.8, 91.0] | 44.4% | 0.0% | 100% |
| **all** | **400** | **90.8% [87.5, 93.2]** | **18.5%** | **0.0%** | **100%** |

**How to read it (honest):**
- **Strongest, cleanest claim:** the model **correctly authorizes 100% of the
  200 genuinely-real, independently-authored in-scope tool calls** (0%
  false-refuse, every domain). It does not wrongly block legitimate real
  agent actions — on tool names and id formats it never trained on.
- On the **unauthorized** side it catches ~81.5% of out-of-scope redirects
  (false-authorize 18.5%), i.e. it transfers resource-scope judgment to real
  vocabulary but **does not beat the format-agnostic heuristic (100%)** on
  this resource-scope task. Report the heuristic floor alongside — the model's
  contribution here is judgment transfer, not out-scoring a glob matcher.
- **Overall 90.8%** on real tau2 tool calls (vs. ~55% under an off-
  distribution representation — see the table above) is the headline transfer
  number, with the representation-sensitivity caveat attached.
- retail's lower 77.8% sits on n=18 (wide CI [54.8, 91.0]); telecom, the
  action-rich domain with n=362, is the reliable estimate at 91.4%.

---

## The fix — representation augmentation + a second independent real source (E5b)

The representation-sensitivity finding above raises the obvious follow-up: is
the brittleness *fundamental*, or just an artifact of training on a single
notation? E5b answers it, and adds a second independent real corpus so the
transfer claim is not a tau2 artifact.

**Method (augmentation).** `augment_representation.py` re-notates each training
trace under a random delimiter scheme — `canonical` (`family:namespace/leaf`),
`allcolon`, `allslash`, `pipe` — applied *consistently* to every resource
string in the trace (root scope, every hop's scope, every action). Because the
verifier's `fnmatch`/subsumption logic special-cases only glob metacharacters
(`*?[]`) and never the delimiters `:`/`/`, a consistent substitution **preserves
every verdict**; the mapper re-verifies with `label_action` and discards
(counting) any trace whose label changes — **0 discarded on 2400 actions**.
CE is retrained on the mixed-notation corpus.

**Data hygiene (held out properly).**
- **train** = `augmented_train` (expanded corpus seed 101, re-notated seed 33).
- **val** = `augmented_val` (expanded corpus seed **202**, re-notated seed 34) —
  a different seed, deduplicated at the decision-context level against both
  train and the committed test; augmented so monitoring reflects the
  mixed-notation distribution. Re-notation changes only the surface, not the
  `(root, chain, action)` decision context dedup was computed on, so it
  introduces no new leakage.
- **test** (never seen in training), two independent real logs:
  - **tau2** balanced confused-deputy set, in both slash (trained) and colon
    (naive) notation.
  - **Toucan** (`Agent-Ark/Toucan-1.5M`, Apache-2.0) — a **second, independent**
    real MCP corpus of thousands of distinct servers (web search, Unity,
    finance, trivia, …). `map_toucan_to_chain.py` maps each trajectory to a
    single-session chain (`sess:*` → `sess:<sid>/*`, specific-tool grants),
    every real `function_call` becoming an **authorized** in-scope action
    (resource `sess:<sid>/<leaf>`, leaf = slug of the call's primary real arg).
    Toucan is good-behavior data, so this is a **recognition test**
    (false-refuse on real legitimate calls), rendered in **mixed native
    notations** via `augment()` — real vocabulary *and* real formats at once.
    An optional `--redirect` flag adds foreign-session confused-deputy negatives
    for a balanced accuracy number.

**What each source proves.** tau2-both-notations = the brittleness is fixed on
the *same* data that exposed it; Toucan-mixed = the model recognizes real
authorized calls from a *different* source across *varied* real notations —
the transfer claim generalizes beyond tau2's three domains and one notation.

**Result (fill after the Colab run `colab_augment.ipynb`, 3 seeds, Wilson CIs):**

| model | committed test | tau2 slash | tau2 colon | Toucan mixed (f-refuse) |
|---|---|---|---|---|
| released CE-0.5B (single notation) | 0.975 | 90.8% | ~55% (brittle) | _pending_ |
| augmented CE-0.5B (mixed notation) | _pending_ | _pending_ | _pending_ | _pending_ |

Claim to make **only if** the augmented model holds up on both tau2 notations
and Toucan: the notation brittleness is **not fundamental** — training across
representations closes it, at no cost to the committed-test accuracy, and
generalizes to an independent real source. If colon/Toucan still lag, report
that honestly as a residual limit with the normalize-at-deployment mitigation.

## Deliverables
- `map_tau_to_chain.py` (+ `test_map_tau_to_chain.py`, offline), `colab_realtrace.ipynb`
- `map_toucan_to_chain.py` (+ `test_map_toucan.py`, offline, 7 tests) — second real source
- `augment_representation.py` (+ `test_augment_representation.py`), `colab_augment.ipynb`
- `real_trace_{telecom,airline,retail,all}.jsonl`, `real_toucan_{all,mixed}.jsonl`
  (gitignored; regenerate from seed 5 / seed 9)
- `results_realtrace.json`, `results_augment.json` (per-test + Wilson CIs)
- AgentDojo (native prompt-injection threat model) is cited as complementary
  future work, not run — a different formalism that would force verdict-vs-label
  reconciliation.
