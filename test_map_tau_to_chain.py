"""
test_map_tau_to_chain.py
=======================

Tests for the tau2 -> chain mapper (E5), on a synthetic tau2-shaped fixture
(no network / no dataset download). Run:

    PYTHONPATH=. python3 test_map_tau_to_chain.py

Validates the parsing/id-extraction logic and — the load-bearing property —
that every emitted label comes from the verifier: authorized = in-scope real
call, unauthorized = the same call redirected outside the served customer's
scope (confused deputy).
"""

import json

from authority_verifier import verify
from map_tau_to_chain import (
    build_traces, call_resource_id, domain_of, ids_in, label_split,
    parse_tool_call, trajectory_calls,
)
from trace_benchmark import trace_to_objects

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


def _asst(content):
    return {"role": "assistant", "content": content}


def _tool(name, obj):
    return {"role": "tool", "name": name, "content": json.dumps(obj)}


# a synthetic tau2-shaped trajectory (telecom-like) + a second (for redirects)
FIX = [
    {"task_id": "[telecom][x]", "response": "", "prompt": [
        {"role": "system", "content": "You are a telecom agent. tools: ..."},
        {"role": "user", "content": "my line is broken, phone 555-1"},
        _asst('<thinking>look up</thinking>\n{"name": "get_customer_by_phone", '
              '"arguments": {"phone_number": "555-1"}}'),
        _tool("get_customer_by_phone",
              {"customer_id": "C1001", "line_ids": ["L1001", "L1002"],
               "bill_ids": ["B1001"]}),
        _asst('<thinking>suspend</thinking>\n{"name": "suspend_line", '
              '"arguments": {"line_id": "L1001"}}'),
    ]},
    {"task_id": "[retail][y]", "response": "", "prompt": [
        {"role": "system", "content": "You are a retail agent."},
        {"role": "user", "content": "cancel my order"},
        _asst('<thinking>details</thinking>\n{"name": "get_order_details", '
              '"arguments": {"order_id": "#W2378156"}}'),
        _tool("get_order_details", {"order_id": "#W2378156", "user_id": "U9"}),
    ]},
]


def test_parse_tool_call_outer_object():
    # the outer object, not the inner "arguments" brace
    name, args = parse_tool_call(
        '<thinking>x</thinking>\n{"name": "suspend_line", '
        '"arguments": {"line_id": "L1001"}}')
    assert name == "suspend_line" and args == {"line_id": "L1001"}
    assert parse_tool_call("<thinking>only thinking</thinking>") is None
    assert parse_tool_call("no json here") is None


def test_id_extraction_domain_agnostic():
    assert call_resource_id({"line_id": "L1001"}) == "L1001"
    assert call_resource_id({"order_id": "#W2378156"}) == "#W2378156"
    assert call_resource_id({"reservation_id": "EHGLP3"}) == "EHGLP3"
    assert call_resource_id({"phone_number": "555-1"}) is None  # lookup, no id
    # plural id-collections are found for the redirect pool
    got = set(ids_in({"customer_id": "C1001", "line_ids": ["L1001", "L1002"]}))
    assert got == {"C1001", "L1001", "L1002"}


def test_domain_of():
    assert domain_of(FIX[0]) == "telecom"
    assert domain_of(FIX[1]) == "retail"


def test_trajectory_calls_skip_lookups():
    calls = trajectory_calls(FIX[0])
    # get_customer_by_phone (no id) skipped; suspend_line (line_id) kept
    assert [c[0] for c in calls] == ["suspend_line"]
    assert calls[0][1] == "L1001"


def test_build_traces_labels_are_verifier_verdicts():
    traces, stats = build_traces(FIX, seed=1)
    assert stats["n_traces"] == 2  # one id-call in each fixture trajectory
    for t in traces:
        root, chain, actions = trace_to_objects(t)
        assert len(actions) == 2  # authorized + redirect
        auth, redir = actions
        # every stored label re-verifies (verifier is ground truth)
        for act, aj in zip(actions, t["actions"]):
            assert (1 if verify(act, chain, root).authorized else 0) == aj["label"]
        # authorized in-scope, redirect out-of-scope
        assert t["actions"][0]["label"] == 1
        assert t["actions"][1]["label"] == 0
        # confused-deputy structure: root covers the redirect resource, the
        # agent's delegated scope does not
        assert root.scope.permits(redir.action, redir.resource, redir.amount)
        assert not chain[-1].scope.permits(redir.action, redir.resource,
                                           redir.amount)


def test_redirect_targets_a_different_trajectory():
    traces, _ = build_traces(FIX, seed=1)
    for t in traces:
        served = t["actions"][0]["resource"].split(":")[1]
        foreign = t["actions"][1]["resource"].split(":")[1]
        assert served != foreign, "redirect must target a foreign namespace"


def test_label_split_and_determinism():
    t1, _ = build_traces(FIX, seed=1)
    t2, _ = build_traces(FIX, seed=1)
    assert t1 == t2
    s = label_split(t1)
    assert s["n_actions"] == 4 and s["authorized"] == 2 and s["unauthorized"] == 2


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
