"""
test_make_novel_domain.py
=========================

Tests for the novel-domain (zero-shot OOD) test-set generator. Run:

    PYTHONPATH=. python3 test_make_novel_domain.py
"""

import trace_benchmark
from authority_verifier import verify
from make_expanded_train import corpus_canonicals
from make_novel_domain import (
    NOVEL_DOMAIN_POOL,
    TRAINING_DOMAIN_PREFIXES,
    TRAINING_RESOURCE_PREFIXES,
    generate_novel,
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


NOVEL = generate_novel(seed=404, traces_per_class=20)


def test_domains_are_novel():
    novel_prefixes = {a["action"].split(".")[0]
                      for tr in NOVEL for a in tr["actions"]}
    assert novel_prefixes == {"calendar", "cloud"}
    assert not (novel_prefixes & TRAINING_DOMAIN_PREFIXES)


def test_pool_prefixes_all_disjoint_from_training():
    for name, d in NOVEL_DOMAIN_POOL.items():
        assert d["pattern"].split(".")[0] not in TRAINING_DOMAIN_PREFIXES, name
        assert d["top"].split(":")[0] not in TRAINING_RESOURCE_PREFIXES, name
    # at least one budgeted domain exists so budget_violation is buildable
    assert any(d["budgeted"] for d in NOVEL_DOMAIN_POOL.values())


def test_multi_domain_selection():
    multi = generate_novel(404, 10, ["iot", "finance", "storage"])
    prefixes = {a["action"].split(".")[0]
                for tr in multi for a in tr["actions"]}
    assert prefixes == {"device", "trade", "blob"}


def test_all_classes_present():
    assert {tr["scenario_class"] for tr in NOVEL} == \
        set(trace_benchmark.SCENARIO_CLASSES)


def test_no_overlap_with_training_domains():
    # regenerate the real 5-domain corpus and confirm zero shared context
    a, b = generate_corpus(101, 30)
    assert not (corpus_canonicals(NOVEL) & corpus_canonicals(a + b))


def test_labels_reverify():
    for tr in NOVEL:
        root, chain, actions = trace_to_objects(tr)
        for act, aj in zip(actions, tr["actions"]):
            assert (1 if verify(act, chain, root).authorized else 0) \
                == aj["label"]


def test_global_domains_restored():
    before = trace_benchmark.DOMAINS
    generate_novel(seed=1, traces_per_class=3)
    assert trace_benchmark.DOMAINS is before, \
        "generate_novel must restore the module DOMAINS global"


def test_deterministic():
    assert generate_novel(404, 20) == NOVEL


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
