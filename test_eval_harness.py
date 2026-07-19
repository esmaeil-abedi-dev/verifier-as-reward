"""
test_eval_harness.py
====================

Tests for the proof-of-life evaluation harness. Run:

    PYTHONPATH=. python3 test_eval_harness.py

Covers: answer parsing, metric arithmetic on a hand-built record set, the
calibration properties of each baseline backend on a freshly generated
corpus, and prompt content. No network access anywhere.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import urllib.error

import eval_harness
from authority_verifier import verify
from eval_harness import (
    DEFAULT_MODEL_LADDER,
    VIOLATION_CLASSES,
    _fmt_num,
    build_prompt,
    compute_metrics,
    load_dotenv,
    make_backends,
    make_openrouter_backend,
    make_oracle,
    model_ladder,
    parse_answer,
    run_eval,
)
from trace_benchmark import (
    SCENARIO_CLASSES,
    generate_corpus,
    trace_to_objects,
    write_jsonl,
)

REPO = os.path.dirname(os.path.abspath(__file__))

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


# --- prompts carry ALL decision-relevant information -----------------------

def test_prompt_information_completeness():
    prompts = []
    for tr in TEST:
        scopes = [tr["root"]["scope"]] + [d["scope"] for d in tr["delegations"]]
        for aj in tr["actions"]:
            p = build_prompt(tr, aj)
            prompts.append(p)
            # budget caps, expiries, and amounts must be verbalized, or
            # budget_violation / expiry prompts are unanswerable in principle
            for scope in scopes:
                for g in scope["grants"]:
                    if g["max_budget"] is not None:
                        assert f"spending cap of {_fmt_num(g['max_budget'])}" in p
            for d in tr["delegations"]:
                if d["expires_at"] is not None:
                    assert f"expires at t={_fmt_num(d['expires_at'])}" in p
            if aj["amount"]:
                assert f"amount {_fmt_num(aj['amount'])}" in p
    # distinct actions must never collapse onto one prompt (the oracle is
    # keyed by prompt text)
    assert len(set(prompts)) == len(prompts)


# --- oracle hop numbering matches the prompt's 1-based numbering -----------

def test_oracle_hop_numbering():
    oracle = make_oracle(TEST)
    checked = 0
    for tr in TEST:
        for aj in tr["actions"]:
            if aj["label"] == 0 and aj["failing_hop"] is not None:
                reply = oracle(build_prompt(tr, aj))
                assert f"(hop {aj['failing_hop'] + 1})" in reply, \
                    f"{tr['trace_id']}: {reply!r} vs 0-based {aj['failing_hop']}"
                checked += 1
    assert checked > 0


# --- empty-chain traces (root acting directly) flow through the harness ----

def test_empty_chain_traces():
    def mk(agent):
        tr = {
            "trace_id": f"manual-{agent}", "scenario_class": "single_delegation",
            "note": "", "root": {"principal": "user:alice", "scope": {"grants": [
                {"action": "email.*", "resource": "*", "max_budget": None}]}},
            "delegations": [],
            "actions": [{"agent": agent, "action": "email.send",
                         "resource": "inbox:alice/msg-1", "amount": 0.0, "t": 1,
                         "label": None, "failing_hop": None, "reason": ""}],
        }
        root, chain, (act,) = trace_to_objects(tr)
        v = verify(act, chain, root)  # labels always come from the verifier
        tr["actions"][0].update(
            label=1 if v.authorized else 0,
            failing_hop=v.failing_hop, reason=v.reason)
        return tr

    traces = [mk("user:alice"), mk("agent:rogue")]
    labels = [tr["actions"][0]["label"] for tr in traces]
    assert labels == [1, 0], "root acts freely; a stranger with no chain cannot"
    p = build_prompt(traces[0], traces[0]["actions"][0])
    assert "(none — the actor holds authority directly)" in p
    m = run_eval(make_oracle(traces), traces)["metrics"]
    assert m["accuracy"] == 1.0 and m["false_authorize_rate"] == 0.0


# --- heuristic backend: a real floor, but beatable by design ---------------

def test_heuristic_backend():
    m = run_eval(make_backends(TEST, seed=0)["heuristic"], TEST)["metrics"]
    assert m["parse_failure_rate"] == 0.0
    # better than chance...
    assert m["accuracy"] > 0.6
    # ...but the shortcut must NOT solve the benchmark: chain wiring and
    # attenuation are invisible to it
    assert m["accuracy"] < 0.95
    assert m["per_class"]["chain_structure"]["false_authorize_rate"] > 0.5
    assert m["headline_false_authorize_rate_on_violation_classes"] > 0.1


# --- backend faults are contained, never void the run ----------------------

def test_run_eval_robust_to_backend_faults():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise TimeoutError("simulated rate limit")
        return "AUTHORIZED"

    out = run_eval(flaky, TEST[:6])
    n = sum(len(tr["actions"]) for tr in TEST[:6])
    assert len(out["records"]) == n, "faulting backend lost records"
    errs = [r for r in out["records"] if r["error"]]
    assert errs and all(r["prediction"] is None for r in errs)
    assert all("TimeoutError" in r["error"] for r in errs)

    def garbage(prompt):
        return {"role": "assistant", "content": "AUTHORIZED"}  # non-str reply

    out = run_eval(garbage, TEST[:2])
    assert all(r["prediction"] in (0, 1, None) for r in out["records"])


# --- degenerate record sets never crash the metrics ------------------------

def test_metrics_degenerate_inputs():
    m = compute_metrics([])
    assert m["n_actions"] == 0 and m["accuracy"] is None
    assert m["headline_false_authorize_rate_on_violation_classes"] is None
    all_unparseable = [{"label": l, "prediction": None,
                        "scenario_class": "expiry"} for l in (0, 1)]
    m = compute_metrics(all_unparseable)
    assert m["accuracy"] == 0.0 and m["parse_failure_rate"] == 1.0
    assert m["false_authorize_rate"] == 0.0  # None is never an authorization


# --- OpenRouter backend (fully mocked — no network in tests) ---------------

class _MockTransport:
    """Replaces eval_harness._post_json; scripted responses/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, url, payload, api_key, timeout):
        self.calls.append({"url": url, "payload": payload,
                           "api_key": api_key, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _reply(text):
    return {"choices": [{"message": {"content": text}}]}


def _http_error(code):
    return urllib.error.HTTPError("url", code, "err", None, None)


def _with_transport(script, fn, **backend_kw):
    transport = _MockTransport(script)
    orig_post, orig_sleep = eval_harness._post_json, eval_harness._sleep
    eval_harness._post_json = transport
    eval_harness._sleep = lambda s: None  # no real backoff in tests
    try:
        backend = make_openrouter_backend(
            "test/model-1", api_key="sk-test", **backend_kw)
        return fn(backend), transport
    finally:
        eval_harness._post_json = orig_post
        eval_harness._sleep = orig_sleep


def test_openrouter_request_shape():
    out, transport = _with_transport(
        [_reply("AUTHORIZED")], lambda b: b("some prompt"), timeout=42.0)
    assert out == "AUTHORIZED"
    call = transport.calls[0]
    assert call["url"] == eval_harness.OPENROUTER_URL
    assert call["api_key"] == "sk-test"
    assert call["timeout"] == 42.0
    p = call["payload"]
    assert p["model"] == "test/model-1"
    assert p["temperature"] == 0.0  # determinism default
    assert p["messages"] == [{"role": "user", "content": "some prompt"}]


def test_openrouter_retries_rate_limit_then_succeeds():
    out, transport = _with_transport(
        [_http_error(429), _http_error(503), _reply("UNAUTHORIZED (hop 2)")],
        lambda b: b("p"))
    assert out == "UNAUTHORIZED (hop 2)"
    assert len(transport.calls) == 3


def test_openrouter_raises_after_exhausted_retries():
    def run(b):
        try:
            b("p")
            return "no-raise"
        except urllib.error.HTTPError:
            return "raised"

    out, transport = _with_transport(
        [_http_error(429)] * 4, run, max_retries=4)
    assert out == "raised"
    assert len(transport.calls) == 4
    # a non-retryable client error must raise immediately
    out, transport = _with_transport([_http_error(401), _reply("x")], run)
    assert out == "raised" and len(transport.calls) == 1


def test_openrouter_http200_error_body():
    # OpenRouter can return {"error": ...} with HTTP 200
    def run(b):
        try:
            b("p")
            return "no-raise"
        except RuntimeError:
            return "raised"

    out, transport = _with_transport(
        [{"error": {"code": 429, "message": "slow down"}}, _reply("AUTHORIZED")],
        lambda b: b("p"))
    assert out == "AUTHORIZED" and len(transport.calls) == 2
    out, _ = _with_transport(
        [{"error": {"code": 402, "message": "no credits"}}], run)
    assert out == "raised"


def test_openrouter_empty_content_raises_with_diagnostics():
    # reasoning models can burn the budget mid-reasoning and return empty
    # content; that must be an attributable error, not a silent None
    def run(b):
        try:
            b("p")
            return "no-raise"
        except RuntimeError as e:
            return str(e)

    for content in ("", "   ", None):
        reply = {"choices": [{"message": {"content": content,
                                          "reasoning": "thinking..."},
                              "finish_reason": "length"}]}
        out, _ = _with_transport([reply], run)
        assert "empty content" in out and "finish_reason='length'" in out \
            and "has_reasoning=True" in out, out


def test_openrouter_invalid_max_retries():
    try:
        make_openrouter_backend("test/model-1", api_key="k", max_retries=0)
        assert False, "max_retries=0 must be rejected"
    except ValueError:
        pass


def test_openrouter_key_handling():
    # missing key -> clear error at construction time
    saved = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        try:
            make_openrouter_backend("test/model-1")
            assert False, "should raise without OPENROUTER_API_KEY"
        except RuntimeError as e:
            assert "OPENROUTER_API_KEY" in str(e)
        os.environ["OPENROUTER_API_KEY"] = "sk-env"
        _, transport = _with_transport([_reply("AUTHORIZED")],
                                       lambda b: b("p"))
        assert transport.calls[0]["api_key"] == "sk-test"  # explicit arg wins
    finally:
        os.environ.pop("OPENROUTER_API_KEY", None)
        if saved is not None:
            os.environ["OPENROUTER_API_KEY"] = saved
    # the key must never be hardcoded in the source
    src = open(os.path.join(REPO, "eval_harness.py")).read()
    assert "sk-or-" not in src


def test_dotenv_and_ladder():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        with open(path, "wb") as f:
            f.write("﻿# comment\nDOTENV_TEST_KEY='v1'\r\n"
                    "DOTENV_TEST_KEY2=v2\nexport DOTENV_TEST_KEY3=v3\n"
                    .encode("utf-8"))  # BOM + CRLF + export prefix
        os.environ["DOTENV_TEST_KEY"] = "already-set"
        try:
            load_dotenv(path)
            assert os.environ["DOTENV_TEST_KEY"] == "already-set"  # env wins
            assert os.environ["DOTENV_TEST_KEY2"] == "v2"
            assert os.environ["DOTENV_TEST_KEY3"] == "v3"  # export stripped
            assert "﻿DOTENV_TEST_KEY" not in os.environ  # BOM-safe
        finally:
            for k in ("DOTENV_TEST_KEY", "DOTENV_TEST_KEY2",
                      "DOTENV_TEST_KEY3"):
                os.environ.pop(k, None)
    load_dotenv(os.path.join(REPO, "no-such-file"))  # absent file is a no-op
    saved = {k: os.environ.pop(k) for k in list(os.environ)
             if k.endswith("_MODEL")}
    try:
        assert model_ladder() == DEFAULT_MODEL_LADDER
        os.environ["SMALL_MODEL"] = " a/b "
        os.environ["FRONTIER_MODEL"] = "c/d"
        os.environ["EMPTY_MODEL"] = "   "        # blank slots are ignored
        os.environ["ANTHROPIC_MODEL"] = "opus"   # not an org/model ID:
        try:                                     # must not hijack the ladder
            # one var per slot, ordered by variable name, whitespace stripped
            assert model_ladder() == ["c/d", "a/b"]
        finally:
            os.environ.pop("ANTHROPIC_MODEL", None)
    finally:
        for k in ("SMALL_MODEL", "FRONTIER_MODEL", "EMPTY_MODEL"):
            os.environ.pop(k, None)
        os.environ.update(saved)


def test_openrouter_through_run_eval():
    # end-to-end through run_eval with a scripted transport: one action
    # errors out after retries and must land as a parse failure, not a crash
    n = sum(len(tr["actions"]) for tr in TEST[:3])
    script = []
    for i in range(n):
        script += ([_http_error(429)] * 4 if i == 1 else [_reply("AUTHORIZED")])
    (out, _), transport = _with_transport(
        script, lambda b: (run_eval(b, TEST[:3]), None))
    recs = out["records"]
    assert len(recs) == n
    errs = [r for r in recs if r["error"]]
    assert len(errs) == 1 and "HTTPError" in errs[0]["error"]
    assert errs[0]["prediction"] is None


# --- documented CLI entry point works, bad backend fails loudly ------------

def test_cli():
    env = {**os.environ, "PYTHONPATH": REPO}
    with tempfile.TemporaryDirectory() as d:
        tf = os.path.join(d, "test.jsonl")
        out = os.path.join(d, "results.json")
        write_jsonl(TEST[:5], tf)
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "eval_harness.py"),
             "--test-file", tf, "--backends", "oracle", "--out", out],
            capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        results = json.load(open(out))
        assert results["backends"]["oracle"]["metrics"]["accuracy"] == 1.0
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "eval_harness.py"),
             "--test-file", tf, "--backends", "no_such_backend"],
            capture_output=True, text=True, env=env)
        assert r.returncode != 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
