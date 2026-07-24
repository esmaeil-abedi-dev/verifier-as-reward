"""
test_make_transfer_val.py
========================

Tests for the synthetic transfer-validation generator. Run:

    PYTHONPATH=. python3 test_make_transfer_val.py

Load-bearing properties: every label is the verifier's; the structure mirrors
the real mapping (single hop, specific-tool grants, foreign-namespace
redirects); vocabulary is synthetic only (no tau2/Toucan leakage); notations
are mixed; generation is deterministic.
"""

from collections import Counter

from authority_verifier import verify
from augment_representation import SCHEMES, augment
from make_transfer_val import FAMILY, TOOLS, build_traces
from trace_benchmark import trace_to_objects

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


TRACES = build_traces(seed=303, n_namespaces=10, calls_per_ns=2)


def test_labels_are_verifier_verdicts_and_balanced():
    labels = []
    for t in TRACES:
        root, chain, actions = trace_to_objects(t)
        for a, aj in zip(actions, t["actions"]):
            assert (1 if verify(a, chain, root).authorized else 0) == aj["label"]
            labels.append(aj["label"])
    assert sum(labels) == len(labels) // 2      # exactly balanced auth/redirect


def test_structure_mirrors_real_mapping():
    for t in TRACES[:8]:
        assert len(t["delegations"]) == 1                       # single hop
        grants = t["delegations"][0]["scope"]["grants"]
        assert all(g["action"] != "*" for g in grants)          # specific tools
        auth, redir = t["actions"]
        assert auth["label"] == 1 and redir["label"] == 0
        ns = auth["resource"].split(":")[1].split("/")[0]
        other = redir["resource"].split(":")[1].split("/")[0]
        assert ns != other                                      # foreign redirect


def test_vocabulary_is_synthetic_only():
    tau2ish = {"suspend_line", "get_reservation_details", "cust"}
    for t in TRACES:
        for a in t["actions"]:
            assert a["action"] in TOOLS
            assert a["resource"].startswith(f"{FAMILY}:")
            assert a["action"] not in tau2ish


def test_mixed_notations_and_label_invariance():
    mixed, discarded = augment(TRACES, seed=304)
    assert discarded == 0
    assert set(Counter(t["_notation"] for t in mixed)) == set(SCHEMES)


def test_deterministic():
    assert build_traces(303, 10, 2) == TRACES


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
