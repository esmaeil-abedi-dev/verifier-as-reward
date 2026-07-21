"""
test_train_verifier_reward.py
=============================

Tests for the verifier-as-reward training harness. Run:

    PYTHONPATH=. python3 test_train_verifier_reward.py

Everything runs on CPU with the offline tiny model — no downloads, no GPU.
Covers: reward correctness against the verifier, prompt/example loading, the
policy scoring path, that a REINFORCE step actually updates the weights and
that a strongly-rewarded decision becomes more likely, and an end-to-end
2-step smoke run that writes the training log.
"""

import argparse
import json
import os
import tempfile

import torch

from authority_verifier import label_action
from trace_benchmark import generate_corpus, write_jsonl
from train_verifier_reward import (
    DECISIONS,
    ByteTokenizer,
    build_model_and_tokenizer,
    candidate_logprobs,
    compact_prompt,
    compute_class_weights,
    evaluate,
    eval_checkpoint,
    load_examples,
    make_checkpoint_backend,
    reward_for_decision,
    train,
)

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


# Shared tiny corpus on disk (what the CLI consumes).
_TMP = tempfile.TemporaryDirectory()
TRAIN_PATH = os.path.join(_TMP.name, "train.jsonl")
TEST_PATH = os.path.join(_TMP.name, "test.jsonl")
_train, _test = generate_corpus(seed=11, traces_per_class=3)
write_jsonl(_train, TRAIN_PATH)
write_jsonl(_test, TEST_PATH)

DEVICE = torch.device("cpu")


def _args(**over):
    base = dict(model="tiny", steps=2, batch_size=2, lr=1e-3, seed=7,
                train_file=TRAIN_PATH, test_file=TEST_PATH, eval_every=1,
                eval_max_actions=8, device="cpu", save_dir=None,
                clip_grad_norm=1.0, entropy_beta=0.0, balance_reward=False,
                exact_pg=False, ce_loss=False, prompt_style="compact",
                warm_start_from=None, kl_coef=0.0,
                log_file=os.path.join(_TMP.name, "training_log.jsonl"))
    base.update(over)
    return argparse.Namespace(**base)


# --- reward comes from the verifier, sign convention ± 1 -------------------

def test_reward_matches_verifier():
    for ex in load_examples(TEST_PATH):
        verdict = label_action(ex["action"], ex["chain"], ex["root"])
        assert reward_for_decision(verdict, ex) == 1.0
        assert reward_for_decision(1 - verdict, ex) == -1.0


# --- example loading -------------------------------------------------------

def test_load_examples():
    exs = load_examples(TEST_PATH)
    assert len(exs) == sum(len(tr["actions"]) for tr in _test)
    for ex in exs:
        p = ex["prompt"]
        assert p.startswith("ROOT ") and p.endswith("DECISION:")
        assert ex["action"].agent in p
        # the label must never leak into the prompt
        assert "label" not in p.lower()


def test_compact_prompt_marks_revocation():
    tr = next(t for t in _test if t["scenario_class"] == "revocation")
    rev_hop = next(d for d in tr["delegations"] if d["revoked_at"] is not None)
    p = compact_prompt(tr, tr["actions"][0])
    assert f"rev={rev_hop['revoked_at']}" in p


# --- tokenizer and policy scoring ------------------------------------------

def test_byte_tokenizer():
    tok = ByteTokenizer()
    ids = tok.encode("ROOT user:alice")
    assert ids and all(0 <= i < 256 for i in ids)
    assert tok.encode("ab") == [97, 98]


def test_candidate_logprobs_shape_and_grad():
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    lps = candidate_logprobs(model, tok, "ROOT x\nDECISION:", DEVICE)
    assert lps.shape == (len(DECISIONS),)
    assert lps.requires_grad
    assert all(float(lp) < 0 for lp in lps)


# --- a policy-gradient step moves probability toward the reward ------------

def test_reinforce_step_updates_policy():
    torch.manual_seed(0)
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    prompt = "ROOT x\nDECISION:"
    before = torch.softmax(
        candidate_logprobs(model, tok, prompt, DEVICE).detach(), dim=0)
    params_before = [p.clone() for p in model.parameters()]
    # reinforce decision index 0 (AUTHORIZE) with reward +1
    for _ in range(5):
        lps = candidate_logprobs(model, tok, prompt, DEVICE)
        loss = -1.0 * torch.log_softmax(lps, dim=0)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = torch.softmax(
        candidate_logprobs(model, tok, prompt, DEVICE).detach(), dim=0)
    assert any(not torch.equal(a, b) for a, b in
               zip(model.parameters(), params_before)), "weights unchanged"
    assert after[0] > before[0], (
        f"P(AUTHORIZE) did not increase: {before[0]} -> {after[0]}")


# --- held-out evaluation metrics -------------------------------------------

def test_evaluate_metrics():
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    exs = load_examples(TEST_PATH)[:8]
    m = evaluate(model, tok, exs, DEVICE)
    assert m["n_eval_actions"] == 8
    assert 0.0 <= m["accuracy"] <= 1.0
    v = m["heldout_violation_rate"]
    assert v is None or 0.0 <= v <= 1.0


# --- end-to-end smoke run: 2 steps, log written ----------------------------

def test_smoke_run_writes_log():
    args = _args()
    history = train(args)
    # step-0 baseline point plus one point per step (eval_every=1)
    assert [h["step"] for h in history] == [0, 1, 2]
    with open(args.log_file) as f:
        logged = [json.loads(line) for line in f]
    # first line records the exact config; the rest are the eval points
    config = logged[0]
    assert config["type"] == "config"
    assert config["args"]["seed"] == args.seed
    assert config["args"]["model"] == "tiny" and config["device"] == "cpu"
    assert logged[1:] == history
    for point in history[1:]:
        assert -1.0 <= point["mean_reward"] <= 1.0
        assert point["cumulative_examples"] == point["step"] * args.batch_size
        assert 0.0 <= point["heldout_violation_rate"] <= 1.0


# --- final checkpoint is saved in HF format when requested -----------------

def test_checkpoint_saving():
    save_dir = os.path.join(_TMP.name, "ckpt")
    train(_args(save_dir=save_dir,
                log_file=os.path.join(_TMP.name, "ckpt_log.jsonl")))
    files = set(os.listdir(save_dir))
    assert "config.json" in files
    assert any(f.endswith((".safetensors", ".bin")) for f in files), files


# --- scores match an independent reference computation ---------------------

def test_candidate_logprobs_numerical_reference():
    # Guards the teacher-forcing offset (start = len(prompt_ids) - 1): an
    # off-by-one would corrupt reward and eval while every shape test passes.
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    model.eval()  # disable dropout so both computations see the same network
    prompt = "ROOT user:alice [email.*|*|inf]\nDECISION:"
    got = candidate_logprobs(model, tok, prompt, DEVICE).detach()
    prompt_ids = tok.encode(prompt)
    for k, cand in enumerate(DECISIONS):
        cand_ids = tok.encode(cand)
        ids = prompt_ids + cand_ids
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        ref = sum(logprobs[i - 1, ids[i]]
                  for i in range(len(prompt_ids), len(ids))) / len(cand_ids)
        assert torch.allclose(got[k], ref, atol=1e-5), \
            f"{cand}: {float(got[k])} != reference {float(ref)}"


# --- untrained policy is not length-biased to one decision -----------------

def test_initial_policy_not_degenerate():
    # Without length normalization the shorter candidate wins with
    # P ~ 1 - 5e-8 and the smoke run learns nothing; pin against regression.
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    exs = load_examples(TEST_PATH)[:5]
    for ex in exs:
        lps = candidate_logprobs(model, tok, ex["prompt"], DEVICE).detach()
        probs = torch.softmax(lps, dim=0)
        assert probs.min() > 0.01, \
            f"untrained policy degenerate: P = {probs.tolist()}"


# --- compact prompts stay well inside the tiny model's context -------------

def test_compact_prompt_length_budget():
    train, test = generate_corpus(seed=7, traces_per_class=25)
    tok = ByteTokenizer()
    limit = 768  # soft bound below n_positions=1024 for early warning
    for tr in train + test:
        for aj in tr["actions"]:
            for cand in DECISIONS:
                n = len(tok.encode(compact_prompt(tr, aj) + cand))
                assert n <= limit, \
                    f"{tr['trace_id']}: {n} tokens exceeds soft limit {limit}"


# --- empty eval set returns clean None metrics, not a crash ----------------

def test_evaluate_empty():
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    m = evaluate(model, tok, [], DEVICE)
    assert m == {"n_eval_actions": 0, "accuracy": None,
                 "heldout_violation_rate": None,
                 "heldout_false_refuse_rate": None}


# --- collapse mitigations: class weights and entropy bonus -----------------

def test_class_weights_and_mitigation_flags():
    from authority_verifier import label_action
    exs = load_examples(TRAIN_PATH)
    w = compute_class_weights(exs)
    labels = [label_action(e["action"], e["chain"], e["root"]) for e in exs]
    n1, n0 = sum(labels), len(labels) - sum(labels)
    # inverse-frequency, normalized: expected weighted count equal per class
    assert abs(w[0] * n0 - w[1] * n1) < 1e-9
    ex1 = exs[labels.index(1)]
    assert reward_for_decision(1, ex1, w) == w[1]
    assert reward_for_decision(0, ex1, w) == -w[1]
    assert reward_for_decision(1, ex1) == 1.0  # unweighted default intact
    # a run with both mitigations on completes and logs normally
    h = train(_args(entropy_beta=0.01, balance_reward=True,
                    log_file=os.path.join(_TMP.name, "mitig_log.jsonl")))
    assert [p["step"] for p in h] == [0, 1, 2]


# --- exact policy gradient: deterministic, learns, no sampling -------------

def test_exact_pg_runs_and_is_deterministic():
    h1 = train(_args(exact_pg=True, balance_reward=True,
                     log_file=os.path.join(_TMP.name, "epg1.jsonl")))
    h2 = train(_args(exact_pg=True, balance_reward=True,
                     log_file=os.path.join(_TMP.name, "epg2.jsonl")))
    assert h1 == h2, "exact-pg must be fully deterministic per seed"
    for p in h1[1:]:
        assert isinstance(p["mean_reward"], float)  # expected reward, smooth


def test_exact_pg_increases_expected_reward():
    # on the tiny model with a healthy lr, the closed-form objective must
    # push the expected verifier reward up within a few steps
    h = train(_args(exact_pg=True, steps=12, batch_size=4, lr=5e-3,
                    eval_every=1,
                    log_file=os.path.join(_TMP.name, "epg3.jsonl")))
    rewards = [p["mean_reward"] for p in h[1:]]
    first, last = sum(rewards[:3]) / 3, sum(rewards[-3:]) / 3
    assert last > first, f"expected reward did not rise: {first} -> {last}"


# --- verifier cross-entropy: converges where exact-PG saturates ------------

def test_ce_loss_drives_train_accuracy_up():
    # CE's non-saturating gradient should let the tiny model FIT the small
    # training corpus — the property exact-PG lacks. Check greedy training
    # accuracy rises substantially over a short high-lr run.
    from authority_verifier import label_action

    def train_acc(model, tok, exs):
        correct = 0
        for ex in exs:
            with torch.no_grad():
                lps = candidate_logprobs(model, tok, ex["prompt"], DEVICE)
            dec = 1 if int(torch.argmax(lps)) == 0 else 0
            correct += int(dec == label_action(ex["action"], ex["chain"],
                                               ex["root"]))
        return correct / len(exs)

    torch.manual_seed(0)
    model, tok = build_model_and_tokenizer("tiny", seed=0)
    model.to(DEVICE)
    exs = load_examples(TRAIN_PATH)
    before = train_acc(model, tok, exs)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    from authority_verifier import label_action as la
    for _ in range(40):
        opt.zero_grad()
        for ex in exs:
            lps = candidate_logprobs(model, tok, ex["prompt"], DEVICE)
            tgt = 0 if la(ex["action"], ex["chain"], ex["root"]) == 1 else 1
            loss = -torch.log_softmax(lps, dim=0)[tgt] / len(exs)
            loss.backward()
        opt.step()
    after = train_acc(model, tok, exs)
    assert after > before + 0.2, f"CE did not fit train: {before} -> {after}"
    assert after >= 0.85, f"CE should nearly fit the tiny corpus: {after}"


def test_ce_loss_flag_runs_and_deterministic():
    h1 = train(_args(ce_loss=True, balance_reward=True,
                     log_file=os.path.join(_TMP.name, "ce1.jsonl")))
    h2 = train(_args(ce_loss=True, balance_reward=True,
                     log_file=os.path.join(_TMP.name, "ce2.jsonl")))
    assert h1 == h2


def test_nl_prompt_style():
    compact = load_examples(TEST_PATH, "compact")
    nl = load_examples(TEST_PATH, "nl")
    assert len(compact) == len(nl)
    # NL prompts are the eval_harness ones (longer, natural language)
    assert nl[0]["prompt"] != compact[0]["prompt"]
    assert "authorization auditor" in nl[0]["prompt"]
    assert compact[0]["prompt"].startswith("ROOT ")


# --- warm-start RL (E1): init from a checkpoint + KL to a frozen ref -------

def test_warm_start_initializes_from_checkpoint():
    # save a (trivially) trained tiny checkpoint, then warm-start from it and
    # confirm the policy begins from those weights, not a fresh base model
    save_dir = os.path.join(_TMP.name, "warm_ckpt")
    train(_args(ce_loss=True, save_dir=save_dir, steps=3, lr=5e-3,
                log_file=os.path.join(_TMP.name, "warm_train.jsonl")))
    from train_verifier_reward import build_model_and_tokenizer
    warm, _ = build_model_and_tokenizer(save_dir, seed=0)
    base, _ = build_model_and_tokenizer("tiny", seed=0)
    warm_sd, base_sd = warm.state_dict(), base.state_dict()
    diff = max(float((warm_sd[k] - base_sd[k]).abs().max())
               for k in warm_sd if warm_sd[k].shape == base_sd[k].shape)
    assert diff > 1e-4, "warm checkpoint should differ from the base model"
    # a warm-started run loads and trains without error
    h = train(_args(warm_start_from=save_dir, balance_reward=True,
                    log_file=os.path.join(_TMP.name, "ws_run.jsonl")))
    assert [p["step"] for p in h] == [0, 1, 2]


def test_kl_zero_when_policy_equals_reference():
    # with kl_coef>0 but policy==ref (both from the same source, step 0), the
    # KL contribution is ~0; a warm+KL run is deterministic and finite
    import torch
    from train_verifier_reward import build_model_and_tokenizer, \
        candidate_logprobs
    m, tok = build_model_and_tokenizer("tiny", seed=0)
    ref, _ = build_model_and_tokenizer("tiny", seed=0)
    ref.eval()
    p = candidate_logprobs(m, tok, "ROOT x\nDECISION:", DEVICE)
    with torch.no_grad():
        rp = candidate_logprobs(ref, tok, "ROOT x\nDECISION:", DEVICE)
    cur = torch.log_softmax(p, 0)
    rlp = torch.log_softmax(rp, 0)
    kl = (cur.exp() * (cur - rlp)).sum()
    assert abs(float(kl)) < 1e-5, f"KL should be ~0 when policy==ref: {kl}"


def test_kl_arm_runs_and_deterministic():
    a = dict(warm_start_from=None, kl_coef=0.1, balance_reward=True)
    h1 = train(_args(log_file=os.path.join(_TMP.name, "kl1.jsonl"), **a))
    h2 = train(_args(log_file=os.path.join(_TMP.name, "kl2.jsonl"), **a))
    assert h1 == h2, "warm+KL REINFORCE must be deterministic per seed"
    for p in h1[1:]:
        assert isinstance(p["mean_reward"], float)


def test_kl_positive_when_policy_diverges_from_reference():
    import torch
    from train_verifier_reward import build_model_and_tokenizer, \
        candidate_logprobs
    m, tok = build_model_and_tokenizer("tiny", seed=0)
    ref, _ = build_model_and_tokenizer("tiny", seed=1)  # different init
    ref.eval()
    cur = torch.log_softmax(
        candidate_logprobs(m, tok, "ROOT x\nDECISION:", DEVICE), 0)
    with torch.no_grad():
        rlp = torch.log_softmax(
            candidate_logprobs(ref, tok, "ROOT x\nDECISION:", DEVICE), 0)
    kl = (cur.exp() * (cur - rlp)).sum()
    assert float(kl) >= -1e-6, "KL is non-negative"


# --- checkpoint ladder-row evaluation (offline, tiny model) ----------------

def test_checkpoint_backend_and_ladder_row():
    import eval_harness as eh
    from authority_verifier import verify as _verify
    from trace_benchmark import trace_to_objects as _tto

    save_dir = os.path.join(_TMP.name, "ckpt_for_eval")
    train(_args(save_dir=save_dir,
                log_file=os.path.join(_TMP.name, "ckpt_eval_log.jsonl")))
    backend = make_checkpoint_backend(save_dir, DEVICE)
    assert backend("ROOT x\nDECISION:") in ("AUTHORIZED", "UNAUTHORIZED")

    # short empty-chain traces (tiny model context is 1024 byte-tokens)
    def mk(agent):
        tr = {"trace_id": f"m-{agent}", "scenario_class": "single_delegation",
              "note": "", "root": {"principal": "u:a", "scope": {"grants": [
                  {"action": "email.*", "resource": "*", "max_budget": None}]}},
              "delegations": [],
              "actions": [{"agent": agent, "action": "email.send",
                           "resource": "inbox:a/m-1", "amount": 0.0, "t": 1,
                           "label": None, "failing_hop": None, "reason": ""}]}
        root, chain, (act,) = _tto(tr)
        v = _verify(act, chain, root)
        tr["actions"][0].update(label=1 if v.authorized else 0,
                                failing_hop=v.failing_hop, reason=v.reason)
        return tr

    traces = [mk("u:a"), mk("a:rogue")]
    tf = os.path.join(_TMP.name, "mini_test.jsonl")
    write_jsonl(traces, tf)
    results_path = os.path.join(_TMP.name, "results.json")
    with open(results_path, "w") as f:
        json.dump({"backends": {}}, f)
    out = eval_checkpoint(save_dir, tf, DEVICE, merge_results=results_path)
    m = out["metrics"]
    assert m["n_actions"] == 2 and m["parse_failure_rate"] == 0.0
    merged = json.load(open(results_path))
    key = f"local:{save_dir}::mini_test.jsonl"  # keyed by model AND test file
    assert merged["backends"][key]["metrics"] == m


# --- same seed reproduces the same training trajectory ---------------------

def test_training_determinism():
    h1 = train(_args(log_file=os.path.join(_TMP.name, "log1.jsonl")))
    h2 = train(_args(log_file=os.path.join(_TMP.name, "log2.jsonl")))
    assert h1 == h2, "identical seeds must reproduce the identical trajectory"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
