"""
test_make_noisy_test.py
======================

Tests for the noise/robustness surrogate (E3). Run:

    PYTHONPATH=. python3 test_make_noisy_test.py

The core guarantee under test: surface noise never changes the verifier label,
and every load-bearing fact survives into the noisy prompt.
"""

import re

from authority_verifier import label_action
from eval_harness import build_prompt, run_eval, make_backends
from make_noisy_test import make_noisy, noisy_prompt_fn
from trace_benchmark import generate_corpus, trace_to_objects

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


_, TEST = generate_corpus(seed=13, traces_per_class=4)
NOISY, DISCARDED = make_noisy(TEST, seed=21)


def test_labels_invariant_under_noise():
    # zero discards, and every noisy label re-verifies to the clean label
    assert DISCARDED == 0
    clean_by = {}
    for tr in TEST:
        root, chain, actions = trace_to_objects(tr)
        clean_by[tr["trace_id"]] = [
            label_action(a, chain, root) for a in actions]
    for tr in NOISY:
        root, chain, actions = trace_to_objects(tr)
        got = [label_action(a, chain, root) for a in actions]
        assert got == clean_by[tr["trace_id"]], tr["trace_id"]
        assert [a["label"] for a in tr["actions"]] == got


def test_structure_unchanged():
    # noise touches only the added noisy_prompt; root/chain/action untouched
    clean_by = {t["trace_id"]: t for t in TEST}
    for tr in NOISY:
        c = clean_by[tr["trace_id"]]
        assert tr["root"] == c["root"]
        assert tr["delegations"] == c["delegations"]
        for na, ca in zip(tr["actions"], c["actions"]):
            for k in ("agent", "action", "resource", "amount", "t", "label"):
                assert na[k] == ca[k]


def test_load_bearing_facts_survive():
    # the action line and every hop line appear verbatim in the noisy prompt
    for tr in NOISY:
        for aj in tr["actions"]:
            noisy = noisy_prompt_fn(tr, aj)
            clean = build_prompt(tr, aj)
            for line in clean.split("\n"):
                if line.startswith(("Hop ", "At time t=", "Root principal:")):
                    assert line in noisy, (tr["trace_id"], line)
            # noise actually changed the prompt and added lines
            assert noisy != clean
            assert len(noisy.split("\n")) > len(clean.split("\n"))


def test_deterministic():
    n2, d2 = make_noisy(TEST, seed=21)
    assert d2 == DISCARDED
    for a, b in zip(NOISY, n2):
        assert a == b


def test_eval_runs_on_noisy_prompts():
    # the oracle (verifier-keyed on the CLEAN prompt) will NOT match noisy
    # prompts, so we use the heuristic which reads prompt text; it should
    # still run end to end and produce metrics with CIs
    out = run_eval(make_backends(NOISY, 0)["heuristic"], NOISY,
                   prompt_fn=noisy_prompt_fn)
    m = out["metrics"]
    assert m["n_actions"] == sum(len(t["actions"]) for t in NOISY)
    assert m["accuracy_ci"] and 0 <= m["accuracy_ci"][0] <= m["accuracy_ci"][1]
    # every record used the noisy prompt (telemetry marker present)
    assert any("[log]" in r["prompt"] or "[trace]" in r["prompt"]
               or "[debug]" in r["prompt"] or "[metrics]" in r["prompt"]
               for r in out["records"])


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
