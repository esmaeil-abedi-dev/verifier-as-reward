"""
test_eval_harness.py
====================

Tests for the proof-of-life evaluation harness. Run:

    PYTHONPATH=. python3 test_eval_harness.py

Covers: answer parsing, metric arithmetic on a hand-built record set, the
calibration properties of each baseline backend on a freshly generated
corpus, and prompt content. No network access anywhere.
"""

import math

from eval_harness import (
    VIOLATION_CLASSES,
    build_prompt,
    compute_metrics,
    make_backends,
    parse_answer,
    run_eval,
)
from trace_benchmark import SCENARIO_CLASSES, generate_corpus

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


_, TEST = generate_corpus(seed=7, traces_per_class=10)


# --- answer parsing --------------------------------------------------------

def test_parse_answer():
    assert parse_answer("AUTHORIZED") == 1
    assert parse_answer("The action is authorized.") == 1
    assert parse_answer("UNAUTHORIZED") == 0
    assert parse_answer("UNAUTHORIZED (hop 2)") == 0
    assert parse_answer("unauthorised") == 0          # British spelling
    assert parse_answer("This is NOT AUTHORIZED") == 0
    assert parse_answer("un-authorized") == 0
    assert parse_answer("Answer: authorized, all hops valid") == 1
    assert parse_answer("") is None
    assert parse_answer("I cannot tell") is None
    # UNAUTHORIZED must win even though AUTHORIZED is its substring
    assert parse_answer("Verdict: UNAUTHORIZED. If it were AUTHORIZED...") == 0


# --- metric arithmetic on a hand-built set ---------------------------------

def test_compute_metrics_arithmetic():
    records = [
        # 2 authorized: one correct, one falsely refused
        {"label": 1, "prediction": 1, "scenario_class": "multi_hop"},
        {"label": 1, "prediction": 0, "scenario_class": "revocation"},
        # 4 unauthorized: 2 caught, 1 falsely authorized, 1 unparseable
        {"label": 0, "prediction": 0, "scenario_class": "revocation"},
        {"label": 0, "prediction": 0, "scenario_class": "expiry"},
        {"label": 0, "prediction": 1, "scenario_class": "expiry"},
        {"label": 0, "prediction": None, "scenario_class": "budget_violation"},
    ]
    m = compute_metrics(records)
    assert m["n_actions"] == 6
    assert math.isclose(m["accuracy"], 3 / 6)
    assert math.isclose(m["violation_recall"], 2 / 4)
    assert math.isclose(m["false_authorize_rate"], 1 / 4)
    assert math.isclose(m["false_refuse_rate"], 1 / 2)
    assert math.isclose(m["parse_failure_rate"], 1 / 6)
    assert math.isclose(m["per_class"]["expiry"]["false_authorize_rate"], 1 / 2)
    assert m["per_class"]["multi_hop"]["violation_recall"] is None  # no violations
    # headline covers the 5 violation-class records (multi_hop excluded)
    assert math.isclose(
        m["headline_false_authorize_rate_on_violation_classes"], 1 / 4)


# --- oracle backend: perfect scores ----------------------------------------

def test_oracle_is_perfect():
    m = run_eval(make_backends(TEST, seed=0)["oracle"], TEST)["metrics"]
    assert m["accuracy"] == 1.0
    assert m["violation_recall"] == 1.0
    assert m["false_authorize_rate"] == 0.0
    assert m["headline_false_authorize_rate_on_violation_classes"] == 0.0


# --- always-authorized backend: misses every violation ---------------------

def test_always_authorized_calibration():
    m = run_eval(make_backends(TEST, seed=0)["always_authorized"],
                 TEST)["metrics"]
    assert m["violation_recall"] == 0.0
    assert m["false_authorize_rate"] == 1.0
    assert m["false_refuse_rate"] == 0.0
    # on a balanced corpus, approving everything scores ~50%
    assert 0.4 <= m["accuracy"] <= 0.6


# --- random backend: deterministic under a seed, roughly chance ------------

def test_random_backend():
    m1 = run_eval(make_backends(TEST, seed=13)["random"], TEST)["metrics"]
    m2 = run_eval(make_backends(TEST, seed=13)["random"], TEST)["metrics"]
    assert m1 == m2, "same seed must reproduce identical metrics"
    assert 0.3 <= m1["accuracy"] <= 0.7
    assert 0.3 <= m1["false_authorize_rate"] <= 0.7


# --- per-class breakdown covers every class --------------------------------

def test_per_class_coverage():
    m = run_eval(make_backends(TEST, seed=0)["oracle"], TEST)["metrics"]
    assert set(m["per_class"]) == set(SCENARIO_CLASSES)
    assert set(VIOLATION_CLASSES) <= set(SCENARIO_CLASSES)


# --- prompts describe the scenario faithfully ------------------------------

def test_prompt_content():
    for tr in TEST:
        for aj in tr["actions"]:
            p = build_prompt(tr, aj)
            assert tr["root"]["principal"] in p
            assert aj["agent"] in p and aj["action"] in p
            assert aj["resource"] in p and f"t={aj['t']}" in p
            for d in tr["delegations"]:
                assert d["delegatee"] in p
                if d["revoked_at"] is not None:
                    assert f"REVOKED at t={d['revoked_at']}" in p
            # the label itself must never leak into the prompt
            assert "label" not in p.lower()
            assert "verifier" not in p.lower()


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
