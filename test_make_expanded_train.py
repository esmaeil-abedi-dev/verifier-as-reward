"""
test_make_expanded_train.py
===========================

Tests for the leakage-guarded expanded-corpus generator. Run:

    PYTHONPATH=. python3 test_make_expanded_train.py
"""

import json
import os
import subprocess
import sys
import tempfile

from authority_verifier import verify
from make_expanded_train import (
    action_canonicals,
    corpus_canonicals,
    drop_overlapping,
)
from trace_benchmark import generate_corpus, load_traces, trace_to_objects, \
    write_jsonl

REPO = os.path.dirname(os.path.abspath(__file__))

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


# --- canonical form captures the decision context, not metadata ------------

def test_canonical_ignores_metadata():
    tr_a, _ = generate_corpus(seed=5, traces_per_class=2)
    tr = tr_a[0]
    import copy
    tr2 = copy.deepcopy(tr)
    tr2["trace_id"] = "renamed-9999"
    tr2["note"] = "different note"
    for a in tr2["actions"]:
        a["label"] = 1 - a["label"]  # labels must not affect identity
        a["reason"] = "x"
    assert action_canonicals(tr) == action_canonicals(tr2)
    # but a changed action time IS a different decision problem
    tr2["actions"][0]["t"] += 1
    assert action_canonicals(tr)[0] != action_canonicals(tr2)[0]


# --- dropping removes exactly the overlapping traces -----------------------

def test_drop_overlapping():
    tr_a, tr_b = generate_corpus(seed=6, traces_per_class=3)
    forbidden = corpus_canonicals(tr_b)
    kept = drop_overlapping(tr_a + tr_b, forbidden, "test")
    assert not (corpus_canonicals(kept) & forbidden)
    assert all(tr in tr_a for tr in kept)


# --- end-to-end CLI: no leakage, labels re-verify, deterministic -----------

def test_cli_end_to_end():
    env = {**os.environ, "PYTHONPATH": REPO}
    with tempfile.TemporaryDirectory() as d:
        test_file = os.path.join(d, "test.jsonl")
        _, test_split = generate_corpus(seed=7, traces_per_class=5)
        write_jsonl(test_split, test_file)
        out_train = os.path.join(d, "tr.jsonl")
        out_val = os.path.join(d, "va.jsonl")
        cmd = [sys.executable, os.path.join(REPO, "make_expanded_train.py"),
               "--train-seed", "101", "--train-traces-per-class", "5",
               "--val-seed", "202", "--val-traces-per-class", "3",
               "--test-file", test_file,
               "--out-train", out_train, "--out-val", out_val]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        train = load_traces(out_train)
        val = load_traces(out_val)
        test_c = corpus_canonicals(test_split)
        assert train and val
        assert not (corpus_canonicals(train) & test_c)
        assert not (corpus_canonicals(val) & test_c)
        assert not (corpus_canonicals(val) & corpus_canonicals(train))
        # every label still reproduces under the verifier
        for tr in train[:30] + val[:30]:
            root, chain, actions = trace_to_objects(tr)
            for act, aj in zip(actions, tr["actions"]):
                assert (1 if verify(act, chain, root).authorized else 0) \
                    == aj["label"]
        # determinism: rerun produces identical bytes
        r2 = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r2.returncode == 0
        assert open(out_train, "rb").read() == \
            open(out_train, "rb").read()  # self-consistent read
        first = open(out_train).read()
        subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert open(out_train).read() == first
        # same seeds must be rejected
        bad = subprocess.run(
            cmd[:2] + ["--train-seed", "9", "--val-seed", "9"],
            capture_output=True, text=True, env=env)
        assert bad.returncode != 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
