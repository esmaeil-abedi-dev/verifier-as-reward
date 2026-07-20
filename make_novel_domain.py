"""
make_novel_domain.py
====================

Generate a test set in BRAND-NEW domains the benchmark has never contained,
for a zero-shot / out-of-distribution probe of an already-trained model — no
retraining. The deployed all-domains model saw email/payment/repo/file/db;
here we hand it `calendar` and `cloud`, with action namespaces and resource
formats it has never seen. If it still scores high, it learned authorization
*structure* (delegation, attenuation, revocation, budgets) that transfers
across surface vocabulary; if it drops, its competence is tied to the
domains it trained on.

Implementation note: the scenario generators in `trace_benchmark.py` draw
their domain from the module-level `DOMAINS` list. This script temporarily
rebinds that list to the novel domains, generates a corpus, then restores it
— so no change to the tested core module is needed, and every trace is still
labeled by the verifier at generation time.

Because the novel domains share NO action prefix or resource prefix with the
five training domains, overlap with any training/test corpus is impossible
by construction (asserted).

Usage:
    PYTHONPATH=. python3 make_novel_domain.py --seed 404 --traces-per-class 40
    # then, with the deployed model:
    PYTHONPATH=. python3 train_verifier_reward.py \
        --eval-checkpoint esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b \
        --test-file novel_domain_test.jsonl

Emits novel_domain_test.jsonl (gitignored; byte-identical from the seed).
"""

from __future__ import annotations

import argparse

import trace_benchmark
from trace_benchmark import generate_corpus, load_traces, write_jsonl
from make_expanded_train import corpus_canonicals, label_stats

# A pool of domains absent from the benchmark's five, same dict shape as the
# built-in DOMAINS entries. Every action prefix and resource prefix here is
# disjoint from the training domains (and from each other). Budgeted domains
# (cloud, finance) let the budget_violation class be exercised.
NOVEL_DOMAIN_POOL = {
    "calendar": {
        "name": "calendar", "pattern": "calendar.*",
        "actions": ["calendar.create", "calendar.read", "calendar.cancel"],
        "top": "cal:*", "namespaces": ["engineering", "sales", "exec"],
        "mid": lambda ns: f"cal:{ns}/*",
        "leaf": lambda rng, ns: f"cal:{ns}/event-{rng.randrange(1000)}",
        "budgeted": False,
    },
    "cloud": {
        "name": "cloud", "pattern": "cloud.*",
        "actions": ["cloud.deploy", "cloud.scale", "cloud.read"],
        "top": "svc:*", "namespaces": ["prod", "staging", "sandbox"],
        "mid": lambda ns: f"svc:{ns}/*",
        "leaf": lambda rng, ns: f"svc:{ns}/inst-{rng.randrange(1000)}",
        "budgeted": True,
    },
    "iot": {
        "name": "iot", "pattern": "device.*",
        "actions": ["device.actuate", "device.read", "device.reset"],
        "top": "dev:*", "namespaces": ["factory", "warehouse", "office"],
        "mid": lambda ns: f"dev:{ns}/*",
        "leaf": lambda rng, ns: f"dev:{ns}/sensor-{rng.randrange(1000)}",
        "budgeted": False,
    },
    "finance": {
        "name": "finance", "pattern": "trade.*",
        "actions": ["trade.execute", "trade.quote", "trade.cancel"],
        "top": "acct:*", "namespaces": ["retail", "institutional", "treasury"],
        "mid": lambda ns: f"acct:{ns}/*",
        "leaf": lambda rng, ns: f"acct:{ns}/order-{rng.randrange(1000)}",
        "budgeted": True,
    },
    "messaging": {
        "name": "messaging", "pattern": "chat.*",
        "actions": ["chat.post", "chat.read", "chat.delete"],
        "top": "channel:*", "namespaces": ["general", "incidents", "leadership"],
        "mid": lambda ns: f"channel:{ns}/*",
        "leaf": lambda rng, ns: f"channel:{ns}/thread-{rng.randrange(1000)}",
        "budgeted": False,
    },
    "storage": {
        "name": "storage", "pattern": "blob.*",
        "actions": ["blob.put", "blob.get", "blob.delete"],
        "top": "bucket:*", "namespaces": ["backups", "media", "logs"],
        "mid": lambda ns: f"bucket:{ns}/*",
        "leaf": lambda rng, ns: f"bucket:{ns}/obj-{rng.randrange(1000)}",
        "budgeted": False,
    },
}

# default selection reproduces the originally-recorded calendar+cloud result
DEFAULT_DOMAINS = ["calendar", "cloud"]

TRAINING_DOMAIN_PREFIXES = {"email", "payment", "repo", "file", "db"}
TRAINING_RESOURCE_PREFIXES = {"inbox", "vendor", "repo", "file", "db"}


def generate_novel(seed: int, traces_per_class: int,
                   domain_names: list = None) -> list:
    """Generate a corpus using only the selected novel domains (both split
    halves merged into one evaluation set). At least one selected domain must
    be budgeted so the budget_violation class can be built."""
    names = domain_names or DEFAULT_DOMAINS
    domains = [NOVEL_DOMAIN_POOL[n] for n in names]
    if not any(d["budgeted"] for d in domains):
        raise SystemExit("select at least one budgeted domain "
                         "(cloud or finance) for budget_violation")
    saved = trace_benchmark.DOMAINS
    try:
        trace_benchmark.DOMAINS = domains
        a, b = generate_corpus(seed, traces_per_class)
    finally:
        trace_benchmark.DOMAINS = saved
    return a + b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--traces-per-class", type=int, default=40)
    ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS),
                    help=f"comma-separated novel domains from "
                         f"{sorted(NOVEL_DOMAIN_POOL)}")
    ap.add_argument("--out", default="novel_domain_test.jsonl")
    args = ap.parse_args()

    names = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = set(names) - set(NOVEL_DOMAIN_POOL)
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; "
                         f"available: {sorted(NOVEL_DOMAIN_POOL)}")
    corpus = generate_novel(args.seed, args.traces_per_class, names)

    # the novel domains must not touch any training-domain vocabulary
    for tr in corpus:
        for a in tr["actions"]:
            assert a["action"].split(".")[0] not in TRAINING_DOMAIN_PREFIXES
            assert a["resource"].split(":")[0] not in TRAINING_RESOURCE_PREFIXES
    write_jsonl(corpus, args.out)

    domains = sorted({a["action"].split(".")[0]
                      for tr in corpus for a in tr["actions"]})
    classes = sorted({tr["scenario_class"] for tr in corpus})
    print(f"novel domains: {domains}")
    print(f"{args.out}: {label_stats(corpus)}")
    print(f"classes ({len(classes)}): {classes}")


if __name__ == "__main__":
    main()
