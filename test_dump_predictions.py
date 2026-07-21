"""
test_dump_predictions.py
=======================

Tests for the per-action prediction dump (E2). Uses offline baseline backends
only — no network, no model download. Run:

    PYTHONPATH=. python3 test_dump_predictions.py
"""

import json
import os
import tempfile

from authority_verifier import verify
from dump_predictions import _structural_features, dump, resolve_backend
from trace_benchmark import generate_corpus, trace_to_objects, write_jsonl

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


_TMP = tempfile.TemporaryDirectory()
_, TEST = generate_corpus(seed=11, traces_per_class=3)
TEST_FILE = os.path.join(_TMP.name, "test.jsonl")
write_jsonl(TEST, TEST_FILE)
N_ACTIONS = sum(len(t["actions"]) for t in TEST)


def test_dump_oracle_full_records():
    out = os.path.join(_TMP.name, "pred_oracle.jsonl")
    dump("oracle", TEST_FILE, out, seed=0)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == N_ACTIONS
    keys = {"backend", "trace_id", "scenario_class", "chain_len", "has_decoy",
            "tight_window", "prompt", "label", "prediction",
            "failing_hop_true", "model_reply_raw"}
    for r in rows:
        assert keys <= set(r)
        assert r["prediction"] in (0, 1, None)
        # oracle = the verifier, so it must be perfect
        assert r["prediction"] == r["label"]
        # unauthorized rows carry a true failing hop; authorized do not
        if r["label"] == 0:
            assert r["failing_hop_true"] is not None
        else:
            assert r["failing_hop_true"] is None


def test_labels_reverify_in_dump():
    # the dumped labels, per trace, must equal a fresh verifier call — no
    # trust in stored labels
    fresh = {}  # trace_id -> multiset (sorted list) of verifier verdicts
    for tr in TEST:
        root, chain, actions = trace_to_objects(tr)
        fresh[tr["trace_id"]] = sorted(
            1 if verify(act, chain, root).authorized else 0 for act in actions)
    out = os.path.join(_TMP.name, "pred_reverify.jsonl")
    dump("oracle", TEST_FILE, out, seed=0)
    dumped = {}
    for r in (json.loads(l) for l in open(out)):
        dumped.setdefault(r["trace_id"], []).append(r["label"])
    for tid, labels in dumped.items():
        assert sorted(labels) == fresh[tid], tid


def test_structural_features():
    feats = _structural_features(TEST)
    assert feats  # non-empty
    for v in feats.values():
        assert set(v) == {"chain_len", "has_decoy", "tight_window"}
        assert isinstance(v["chain_len"], int)
        assert isinstance(v["has_decoy"], bool)
        assert isinstance(v["tight_window"], bool)


def test_resolve_backend_errors_on_unknown():
    try:
        resolve_backend("no_such_backend", TEST, 0)
        assert False, "unknown backend should raise"
    except SystemExit:
        pass


def test_heuristic_dump_matches_metrics():
    # the per-row predictions must aggregate to the same accuracy the
    # evaluator reports (dump and eval share run_eval)
    from eval_harness import make_heuristic, run_eval
    out = os.path.join(_TMP.name, "pred_h2.jsonl")
    dump("heuristic", TEST_FILE, out, seed=0)
    rows = [json.loads(l) for l in open(out)]
    acc_dump = sum(r["prediction"] == r["label"] for r in rows) / len(rows)
    m = run_eval(make_heuristic(), TEST)["metrics"]
    assert abs(acc_dump - m["accuracy"]) < 1e-9


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
