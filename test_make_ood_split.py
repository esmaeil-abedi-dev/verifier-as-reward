"""
test_make_ood_split.py
======================

Tests for the OOD domain-hold-out split generator. Run:

    PYTHONPATH=. python3 test_make_ood_split.py
"""

import os
import subprocess
import sys
import tempfile

from authority_verifier import verify
from make_expanded_train import corpus_canonicals
from make_ood_split import partition_by_domain
from trace_benchmark import (
    generate_corpus,
    load_traces,
    trace_domain,
    trace_to_objects,
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


A, B = generate_corpus(seed=303, traces_per_class=20)
CORPUS = A + B


def test_traces_single_domain():
    for tr in CORPUS:
        doms = {a["action"].split(".")[0] for a in tr["actions"]}
        assert len(doms) == 1, f"{tr['trace_id']} spans {doms}"


def test_partition_disjoint_domains():
    train = partition_by_domain(CORPUS, {"email", "payment", "repo"})
    test = partition_by_domain(CORPUS, {"file", "db"})
    assert train and test
    assert {trace_domain(t) for t in train} <= {"email", "payment", "repo"}
    assert {trace_domain(t) for t in test} <= {"file", "db"}
    # no shared domain => no shared decision context
    assert not (corpus_canonicals(train) & corpus_canonicals(test))
    # partition is exhaustive over the two domain sets
    assert len(train) + len(test) == len(CORPUS)


def test_ood_test_actions_are_novel():
    train = partition_by_domain(CORPUS, {"email", "payment", "repo"})
    test = partition_by_domain(CORPUS, {"file", "db"})
    train_actions = {a["action"] for tr in train for a in tr["actions"]}
    train_res_prefixes = {a["resource"].split(":")[0]
                          for tr in train for a in tr["actions"]}
    for tr in test:
        for a in tr["actions"]:
            assert a["action"] not in train_actions, \
                f"OOD action {a['action']} seen in train"
            assert a["resource"].split(":")[0] not in train_res_prefixes, \
                f"OOD resource family {a['resource']} seen in train"


def test_cli_and_labels_reverify():
    env = {**os.environ, "PYTHONPATH": REPO}
    with tempfile.TemporaryDirectory() as d:
        out_tr = os.path.join(d, "tr.jsonl")
        out_te = os.path.join(d, "te.jsonl")
        cmd = [sys.executable, os.path.join(REPO, "make_ood_split.py"),
               "--seed", "303", "--traces-per-class", "10",
               "--out-train", out_tr, "--out-test", out_te]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        train, test = load_traces(out_tr), load_traces(out_te)
        assert train and test
        # labels re-verify under the verifier
        for tr in (train[:20] + test[:20]):
            root, chain, actions = trace_to_objects(tr)
            for act, aj in zip(actions, tr["actions"]):
                assert (1 if verify(act, chain, root).authorized else 0) \
                    == aj["label"]
        # determinism
        first = open(out_tr).read()
        subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert open(out_tr).read() == first
        # overlapping domains rejected
        bad = subprocess.run(
            cmd + ["--train-domains", "email,file", "--test-domains", "file"],
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
