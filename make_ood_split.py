"""
make_ood_split.py
=================

Build an OUT-OF-DISTRIBUTION domain-hold-out split for the generalization /
overfitting check: train on some domains, test on entirely different ones.

Because the trace generator draws each trace from a single domain (email,
payment, repo, file, db) with a domain-specific action namespace and resource
format, holding out whole domains gives a test set whose actions (`file.read`,
`db.query`) and resources (`file:/projects/...`, `db:...`) never appeared in
training. A model that scores high here learned the authorization *rule*; one
that only learned the generator's distribution will drop.

Default split: train = email, payment, repo; test = file, db. Payment is kept
in training on purpose so the payment-only `budget_violation` class is learned
(the OOD test therefore covers 8 of the 9 classes — disclose this).

Leakage: train and test share NO domain by construction, so no decision
context can overlap; the script still asserts zero (root, chain, action)
overlap as a guard, and labels are re-verifiable (every label came from
`verify(...)` at generation time).

Usage:
    PYTHONPATH=. python3 make_ood_split.py \
        --seed 303 --traces-per-class 200 \
        --train-domains email,payment,repo --test-domains file,db

Emits ood_train.jsonl and ood_test.jsonl (gitignored; byte-identical from
the seed).
"""

from __future__ import annotations

import argparse
import json

from trace_benchmark import (
    corpus_domains_present,
    generate_corpus,
    load_traces,
    trace_domain,
    write_jsonl,
)
from make_expanded_train import corpus_canonicals, label_stats


def partition_by_domain(traces: list, domains: set) -> list:
    return [tr for tr in traces if trace_domain(tr) in domains]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--traces-per-class", type=int, default=200)
    ap.add_argument("--train-domains", default="email,payment,repo")
    ap.add_argument("--test-domains", default="file,db")
    ap.add_argument("--out-train", default="ood_train.jsonl")
    ap.add_argument("--out-test", default="ood_test.jsonl")
    args = ap.parse_args()

    train_domains = {d.strip() for d in args.train_domains.split(",")}
    test_domains = {d.strip() for d in args.test_domains.split(",")}
    if train_domains & test_domains:
        raise SystemExit(f"train and test domains overlap: "
                         f"{train_domains & test_domains}")

    a, b = generate_corpus(args.seed, args.traces_per_class)
    corpus = a + b
    present = corpus_domains_present(corpus)
    unknown = (train_domains | test_domains) - present
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; present: {present}")

    train = partition_by_domain(corpus, train_domains)
    test = partition_by_domain(corpus, test_domains)

    # hard guards: no shared domain, no shared decision context
    assert {trace_domain(t) for t in train} <= train_domains
    assert {trace_domain(t) for t in test} <= test_domains
    assert not (corpus_canonicals(train) & corpus_canonicals(test)), \
        "OOD train/test decision-context overlap"

    write_jsonl(train, args.out_train)
    write_jsonl(test, args.out_test)
    print(f"train domains {sorted(train_domains)}: {label_stats(train)}")
    print(f"test  domains {sorted(test_domains)}: {label_stats(test)}")
    tr_cls = sorted({t["scenario_class"] for t in train})
    te_cls = sorted({t["scenario_class"] for t in test})
    print(f"train classes ({len(tr_cls)}): {tr_cls}")
    print(f"test  classes ({len(te_cls)}): {te_cls}")
    missing = set(tr_cls) - set(te_cls)
    if missing:
        print(f"NOTE: classes absent from OOD test (domain-bound): {missing}")


if __name__ == "__main__":
    main()
