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
    assert merged["backends"][f"local:{save_dir}"]["metrics"] == m


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
