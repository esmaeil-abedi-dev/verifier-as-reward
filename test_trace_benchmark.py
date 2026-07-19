"""
test_trace_benchmark.py
=======================

Tests for the labeled trace corpus generator. Run:

    PYTHONPATH=. python3 test_trace_benchmark.py

Covers: label balance, presence of all 8 scenario classes, no trace leakage
across the train/test split, verifier-reproducibility of every stored label,
seed determinism, and serialization round-tripping.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

from authority_verifier import verify
from trace_benchmark import (
    DATASHEET_FILE,
    SCENARIO_CLASSES,
    TEST_FILE,
    TRAIN_FILE,
    delegation_from_json,
    delegation_to_json,
    generate_corpus,
    load_traces,
    scope_from_json,
    scope_to_json,
    trace_to_objects,
    write_datasheet,
    write_jsonl,
)

REPO = os.path.dirname(os.path.abspath(__file__))

SEED = 7
TRACES_PER_CLASS = 25

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


TRAIN, TEST = generate_corpus(SEED, TRACES_PER_CLASS)
ALL = TRAIN + TEST


def all_actions(traces):
    return [(tr, a) for tr in traces for a in tr["actions"]]


# --- label balance ---------------------------------------------------------

def test_label_balance():
    for name, split in (("train", TRAIN), ("test", TEST)):
        labels = [a["label"] for _, a in all_actions(split)]
        frac = sum(labels) / len(labels)
        assert 0.4 <= frac <= 0.6, f"{name} authorized fraction {frac:.3f}"


# --- all classes present in both splits ------------------------------------

def test_all_classes_present():
    for name, split in (("train", TRAIN), ("test", TEST)):
        present = {tr["scenario_class"] for tr in split}
        assert present == set(SCENARIO_CLASSES), \
            f"{name} missing {set(SCENARIO_CLASSES) - present}"


# --- no trace leakage across splits ----------------------------------------

def test_no_leakage():
    train_ids = {tr["trace_id"] for tr in TRAIN}
    test_ids = {tr["trace_id"] for tr in TEST}
    assert len(train_ids) == len(TRAIN), "duplicate trace_id in train"
    assert len(test_ids) == len(TEST), "duplicate trace_id in test"
    assert not (train_ids & test_ids), f"leaked: {train_ids & test_ids}"


# --- every stored label reproduces under the verifier ----------------------

def test_labels_reproduce_under_verifier():
    n = 0
    for tr in ALL:
        root, chain, actions = trace_to_objects(tr)
        for act, stored in zip(actions, tr["actions"]):
            verdict = verify(act, chain, root)
            assert (1 if verdict.authorized else 0) == stored["label"], \
                f"{tr['trace_id']}: stored {stored['label']} != verifier"
            if not verdict.authorized:
                assert verdict.failing_hop == stored["failing_hop"], \
                    f"{tr['trace_id']}: failing_hop mismatch"
                assert verdict.reason == stored["reason"], \
                    f"{tr['trace_id']}: reason mismatch"
            n += 1
    assert n > 0


# --- unauthorized actions carry hop + reason; authorized carry neither -----

def test_verdict_metadata_consistency():
    for tr, a in all_actions(ALL):
        if a["label"] == 0:
            assert a["reason"], f"{tr['trace_id']}: unauthorized without reason"
        else:
            assert a["failing_hop"] is None and a["reason"] == ""


# --- same seed regenerates byte-identical files ----------------------------

def test_seed_determinism():
    train2, test2 = generate_corpus(SEED, TRACES_PER_CLASS)
    assert train2 == TRAIN and test2 == TEST, "regeneration differs"
    train3, _ = generate_corpus(SEED + 1, TRACES_PER_CLASS)
    assert train3 != TRAIN, "different seed produced identical corpus"
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
        write_jsonl(TRAIN, p1)
        write_jsonl(train2, p2)
        assert open(p1, "rb").read() == open(p2, "rb").read()


# --- serialization round-trips (inf <-> null) ------------------------------

def test_serialization_roundtrip():
    for tr in ALL[:20]:
        for d in tr["delegations"]:
            assert d == delegation_to_json(delegation_from_json(d))
        s = tr["root"]["scope"]
        assert s == scope_to_json(scope_from_json(s))
    # inf never appears in the JSON payload
    for tr in ALL:
        json.dumps(tr, allow_nan=False)
    # and null comes back as inf
    root, chain, _ = trace_to_objects(ALL[0])
    assert math.isinf(root.scope.grants[0].max_budget)


# --- emitted files parse back to the same corpus ---------------------------

def test_jsonl_files_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        write_jsonl(TEST, path)
        assert load_traces(path) == TEST


# --- per-class label shape (violation classes actually violate) ------------

def test_class_label_shapes():
    for tr in ALL:
        labels = [a["label"] for a in tr["actions"]]
        cls = tr["scenario_class"]
        if cls in ("single_delegation", "multi_hop"):
            assert all(l == 1 for l in labels), f"{tr['trace_id']}"
        elif cls == "scope_escalation":
            assert all(l == 0 for l in labels), f"{tr['trace_id']}"
        else:
            assert 0 in labels and 1 in labels, f"{tr['trace_id']}"


# --- committed artifacts match regeneration (no silent drift) --------------

def test_committed_artifacts_match_regeneration():
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(TRAIN, os.path.join(d, TRAIN_FILE))
        write_jsonl(TEST, os.path.join(d, TEST_FILE))
        write_datasheet(TRAIN, TEST, SEED, TRACES_PER_CLASS,
                        os.path.join(d, DATASHEET_FILE))
        for name in (TRAIN_FILE, TEST_FILE, DATASHEET_FILE):
            committed = os.path.join(REPO, name)
            assert os.path.exists(committed), \
                f"{name} missing — run trace_benchmark.py first"
            assert (open(committed, "rb").read()
                    == open(os.path.join(d, name), "rb").read()), \
                f"committed {name} does not match regeneration with " \
                f"seed={SEED}, traces_per_class={TRACES_PER_CLASS}"


# --- generator intent asserts hold across many seeds -----------------------

def test_many_seed_robustness():
    # make_trace asserts generator intent == verifier verdict on every
    # action; boundary draws (t == revoked_at etc.) vary by seed, so sweep.
    for s in range(30):
        generate_corpus(seed=s, traces_per_class=5)


# --- too-small corpus is rejected, not silently degenerate -----------------

def test_tiny_corpus_rejected():
    try:
        generate_corpus(seed=1, traces_per_class=1)
        assert False, "traces_per_class=1 should raise"
    except ValueError:
        pass
    train2, test2 = generate_corpus(seed=1, traces_per_class=2)
    for split in (train2, test2):
        assert {tr["scenario_class"] for tr in split} == set(SCENARIO_CLASSES)


# --- scope escalations are diverse (not one canonical widening) ------------

def test_escalation_diversity():
    dims = set()
    for tr in ALL:
        if tr["scenario_class"] == "scope_escalation":
            dims.add(tr["note"].split("widened the ")[1].split()[0])
    assert len(dims) >= 2, f"only escalation dims {dims} in the corpus"


# --- confused-deputy invariant is structural, not vacuous ------------------

def test_confused_deputy_invariant():
    for tr in ALL:
        if tr["scenario_class"] != "attack_confused_deputy":
            continue
        root, chain, actions = trace_to_objects(tr)
        grants = root.scope.grants
        assert not any(g.action == "*" and g.resource == "*" for g in grants), \
            f"{tr['trace_id']}: deputy root must not hold universal authority"
        for act, aj in zip(actions, tr["actions"]):
            if aj["label"] == 0:
                # the root could authorize what the deputy's chain cannot
                assert root.scope.permits(act.action, act.resource, act.amount), \
                    f"{tr['trace_id']}: root cannot cover the victim resource"


# --- documented CLI entry point works ---------------------------------------

def test_cli():
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "PYTHONPATH": REPO}
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "trace_benchmark.py"),
             "--seed", "3", "--traces-per-class", "2", "--outdir", d],
            capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        for name in (TRAIN_FILE, TEST_FILE, DATASHEET_FILE):
            assert os.path.exists(os.path.join(d, name)), name
        assert load_traces(os.path.join(d, TRAIN_FILE))


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
