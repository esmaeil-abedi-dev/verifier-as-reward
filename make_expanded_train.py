"""
make_expanded_train.py
======================

Generate an EXPANDED training corpus plus a separate VALIDATION corpus for
verifier-as-reward fine-tuning, with example-level leakage guards against
the committed benchmark test set.

Leakage discipline (journal requirement):
  - The committed benchmark_test.jsonl is never regenerated or touched; it
    is the one held-out set, evaluated once at the end.
  - The expanded corpora come from DIFFERENT generator seeds. Because the
    generator draws from a finite vocabulary, identical decision contexts
    can recur across seeds — so every trace whose (root, chain, action)
    canonical form collides with ANY test-set example is dropped. The
    validation corpus is additionally deduplicated against the expanded
    train corpus.
  - Validation is for tuning/monitoring; the committed test set is for the
    final reported numbers only.

Usage:
    PYTHONPATH=. python3 make_expanded_train.py \
        --train-seed 101 --train-traces-per-class 150 \
        --val-seed 202 --val-traces-per-class 25 \
        --test-file benchmark_test.jsonl

Emits expanded_train.jsonl and expanded_val.jsonl (gitignored; regenerate
anywhere with the same seeds — byte-identical).
"""

from __future__ import annotations

import argparse
import json

from trace_benchmark import generate_corpus, load_traces, write_jsonl


def action_canonicals(trace: dict) -> list:
    """One canonical string per action: the exact decision context
    (root, full chain, action core) with labels/ids/notes stripped. Two
    examples with equal canonicals are the same decision problem."""
    context = {
        "root": trace["root"],
        "delegations": trace["delegations"],
    }
    out = []
    for a in trace["actions"]:
        core = {k: a[k] for k in ("agent", "action", "resource", "amount", "t")}
        out.append(json.dumps({**context, "action": core}, sort_keys=True))
    return out


def corpus_canonicals(traces: list) -> set:
    return {c for tr in traces for c in action_canonicals(tr)}


def drop_overlapping(traces: list, forbidden: set, name: str) -> list:
    kept = []
    dropped = 0
    for tr in traces:
        if any(c in forbidden for c in action_canonicals(tr)):
            dropped += 1
        else:
            kept.append(tr)
    print(f"{name}: kept {len(kept)} traces, dropped {dropped} overlapping")
    return kept


def label_stats(traces: list) -> str:
    labels = [a["label"] for tr in traces for a in tr["actions"]]
    return (f"{len(labels)} actions, "
            f"{sum(labels)} authorized / {len(labels) - sum(labels)} "
            f"unauthorized ({sum(labels) / len(labels):.1%} authorized)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--train-seed", type=int, default=101)
    ap.add_argument("--train-traces-per-class", type=int, default=150)
    ap.add_argument("--val-seed", type=int, default=202)
    ap.add_argument("--val-traces-per-class", type=int, default=25)
    ap.add_argument("--test-file", default="benchmark_test.jsonl")
    ap.add_argument("--out-train", default="expanded_train.jsonl")
    ap.add_argument("--out-val", default="expanded_val.jsonl")
    args = ap.parse_args()

    if args.train_seed == args.val_seed:
        raise SystemExit("train and val seeds must differ")

    test_canon = corpus_canonicals(load_traces(args.test_file))
    print(f"test set: {len(test_canon)} protected decision contexts")

    # both halves of each generated corpus are used — the split that matters
    # here is against the committed test set, not within the new draw
    tr_a, tr_b = generate_corpus(args.train_seed,
                                 args.train_traces_per_class)
    train = drop_overlapping(tr_a + tr_b, test_canon, "expanded train")

    va, vb = generate_corpus(args.val_seed, args.val_traces_per_class)
    val = drop_overlapping(va + vb,
                           test_canon | corpus_canonicals(train),
                           "validation")

    # hard post-conditions: zero cross-set overlap
    train_canon = corpus_canonicals(train)
    val_canon = corpus_canonicals(val)
    assert not (train_canon & test_canon), "train/test leakage"
    assert not (val_canon & test_canon), "val/test leakage"
    assert not (val_canon & train_canon), "val/train leakage"

    write_jsonl(train, args.out_train)
    write_jsonl(val, args.out_val)
    print(f"{args.out_train}: {label_stats(train)}")
    print(f"{args.out_val}: {label_stats(val)}")


if __name__ == "__main__":
    main()
