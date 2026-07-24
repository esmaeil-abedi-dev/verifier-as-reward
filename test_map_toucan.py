"""
test_map_toucan.py
=================

Tests for the Toucan real-trace mapper. Run:

    PYTHONPATH=. python3 test_map_toucan.py

These exercise the parsing/labeling on SYNTHETIC message fixtures (no network),
plus the load-bearing property shared with the augmentation experiment: every
mapped label is the verifier's, and re-notating the resource never changes it.
"""

import json

from authority_verifier import verify
from augment_representation import SCHEMES, label_preserved, renotate_trace
from map_toucan_to_chain import (
    _clean_tool, _primary_arg, _slug, build_traces, trajectory_calls,
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


def _msg_call(name, arguments):
    """An assistant tool-call turn in Toucan's shape (arguments as JSON string)."""
    return {"role": "assistant", "content": "",
            "function_call": {"name": name, "arguments": json.dumps(arguments)}}


# a small, realistic multi-server fixture (two sessions)
FIXTURE = [
    ("uuid-a", [
        {"role": "user", "content": "research StellarPay"},
        {"role": "assistant", "content": "on it"},
        _msg_call("exa-search-company_research_exa", {"query": "StellarPay fintech"}),
        {"role": "function", "name": "exa-search-company_research_exa",
         "content": "{\"results\": []}"},
        _msg_call("exa-search-linkedin_search_exa", {"name": "StellarPay CTO"}),
    ]),
    ("uuid-b", [
        {"role": "user", "content": "unity lighting"},
        _msg_call("unity-mcp-verify_connection", {}),        # no args -> fallback leaf
        {"role": "function", "name": "unity-mcp-verify_connection", "content": "{}"},
        _msg_call("unity-mcp-read_file", {"path": "Assets/Light.cs"}),
    ]),
]


def test_slug_and_primary_arg():
    assert _slug("Assets/Light.cs", "x") == "assets-light-cs"
    assert _slug(None, "call3") == "call3"          # missing arg -> fallback
    assert _slug("", "call3") == "call3"
    assert ":" not in _slug("a:b/c", "x") and "/" not in _slug("a:b/c", "x")
    assert _primary_arg({"query": "hello world"}) == "hello world"
    assert _primary_arg({"foo": {"deep": 1}, "id": "X9"}) == "X9"   # id-hint wins
    assert _primary_arg({}) is None


def test_clean_tool_delimiters():
    # tool names must not carry ':'/'/' (would break renotation of the trace)
    assert _clean_tool("srv:tool/x") == "srv-tool-x"


def test_trajectory_calls_parse():
    calls = trajectory_calls(FIXTURE[0][1])
    assert [c[0] for c in calls] == [
        "exa-search-company_research_exa", "exa-search-linkedin_search_exa"]
    assert calls[0][1] == "stellarpay-fintech"       # slug of the query arg
    # the no-arg unity call gets a fallback leaf, not an empty/'none' leaf
    ucalls = trajectory_calls(FIXTURE[1][1])
    assert ucalls[0][1].startswith("call")


def test_authorized_only_all_authorized():
    traces, stats = build_traces(FIXTURE, seed=1, redirect=False)
    assert stats["n_calls_extracted"] == 4       # 2 calls in each of 2 sessions
    labels = [a["label"] for t in traces for a in t["actions"]]
    assert labels == [1, 1, 1, 1], labels        # every real in-scope call authorized
    # each action's stored label really is a fresh verifier verdict
    for t in traces:
        root, chain, actions = trace_to_objects(t)
        for a, aj in zip(actions, t["actions"]):
            assert (1 if verify(a, chain, root).authorized else 0) == aj["label"]


def test_redirect_is_balanced_and_unauthorized():
    traces, _ = build_traces(FIXTURE, seed=1, redirect=True)
    for t in traces:
        assert t["actions"][0]["label"] == 1        # authorized in-scope
        if len(t["actions"]) > 1:
            assert t["actions"][1]["label"] == 0     # foreign redirect unauthorized


def test_labels_invariant_under_renotation():
    # the property the augmentation experiment relies on, on Toucan traces too
    traces, _ = build_traces(FIXTURE, seed=1, redirect=True)
    for scheme in SCHEMES:
        for t in traces:
            aug = renotate_trace(t, scheme)
            assert label_preserved(aug), (scheme, t["trace_id"])
            root, chain, actions = trace_to_objects(aug)
            for a, aj in zip(actions, aug["actions"]):
                assert (1 if verify(a, chain, root).authorized else 0) == aj["label"]


def test_deterministic():
    a, _ = build_traces(FIXTURE, seed=7, redirect=True)
    b, _ = build_traces(FIXTURE, seed=7, redirect=True)
    assert a == b


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
