"""
test_augment_representation.py
=============================

Tests for representation augmentation. Run:

    PYTHONPATH=. python3 test_augment_representation.py

The load-bearing property: re-notating resources never changes a verifier
verdict (so the augmented labels are still the verifier's), and every scheme
is actually applied.
"""

from authority_verifier import verify
from augment_representation import (
    SCHEMES, augment, label_preserved, renotate_resource, renotate_trace,
)
from trace_benchmark import generate_corpus, trace_to_objects

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


_, TEST = generate_corpus(seed=17, traces_per_class=4)


def test_renotate_resource():
    assert renotate_resource("inbox:alice/msg-1", "canonical") == "inbox:alice/msg-1"
    assert renotate_resource("inbox:alice/msg-1", "allcolon") == "inbox:alice:msg-1"
    assert renotate_resource("inbox:alice/msg-1", "allslash") == "inbox/alice/msg-1"
    assert renotate_resource("inbox:alice/msg-1", "pipe") == "inbox|alice|msg-1"
    # globs and bare '*' preserved
    assert renotate_resource("inbox:alice/*", "allcolon") == "inbox:alice:*"
    assert renotate_resource("*", "pipe") == "*"


def test_labels_invariant_all_schemes():
    # every scheme, every trace: the verifier verdict is unchanged
    for scheme in SCHEMES:
        for tr in TEST:
            aug = renotate_trace(tr, scheme)
            root, chain, actions = trace_to_objects(aug)
            for a, aj in zip(actions, aug["actions"]):
                assert (1 if verify(a, chain, root).authorized else 0) == aj["label"], \
                    (scheme, aug["trace_id"])
            assert label_preserved(aug)


def test_renotation_is_consistent_within_trace():
    # scope globs and action resources use the SAME delimiters (else fnmatch
    # would break) — check no canonical delimiter survives in a non-canonical
    # scheme's leaf separator
    for tr in TEST[:10]:
        aug = renotate_trace(tr, "allcolon")
        for a in aug["actions"]:
            assert "/" not in a["resource"]  # slash replaced by colon
        for d in aug["delegations"]:
            for g in d["scope"]["grants"]:
                assert "/" not in g["resource"]


def test_augment_mixes_and_preserves():
    aug, discarded = augment(TEST, seed=5)
    assert discarded == 0, "re-notation must never flip a label"
    assert len(aug) == len(TEST)
    schemes_used = {t["_notation"] for t in aug}
    assert schemes_used == set(SCHEMES), schemes_used  # all schemes appear
    # deterministic
    aug2, _ = augment(TEST, seed=5)
    assert aug == aug2


def test_structure_otherwise_unchanged():
    # only resource strings change; actions/timing/principals identical
    by = {t["trace_id"]: t for t in TEST}
    for aug in [renotate_trace(t, "allcolon") for t in TEST[:8]]:
        orig = by[aug["trace_id"]]
        assert aug["root"]["principal"] == orig["root"]["principal"]
        for na, oa in zip(aug["actions"], orig["actions"]):
            for k in ("agent", "action", "amount", "t", "label"):
                assert na[k] == oa[k]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
