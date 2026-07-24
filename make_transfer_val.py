"""
make_transfer_val.py
===================

A synthetic *transfer-validation* set that mirrors the STRUCTURE of the
real-trace mapping (single-hop, specific-tool grants, namespace-scoped
confused-deputy redirects, mixed notations) using ONLY synthetic vocabulary.

Why (journey lesson). Across every training variant, the 9-class synthetic
validation saturated at ~96-100% while real-tau2 transfer swung 50-91% — the
standard validation does not predict real transfer, because its structure
(multi-hop, 9 scenario classes, canonical notation) does not match the real
test's structure (single-hop confused deputy over namespaces, varied
notation). This set closes that gap for *checkpoint selection only*: it is
built from synthetic tools/ids (no tau2/Toucan vocabulary — those corpora
stay fully held out) and is never trained on.

Structure per trace (exactly the real-mapping shape):
  root  = system holds Grant("*", "<fam>:*")
  hop   = system -> agent, SPECIFIC-tool grants on "<fam>:<ns>/*"
  auth  = real-shaped in-scope action  "<fam>:<ns>/<id>"      (verifier: 1)
  redir = same tool on a FOREIGN namespace "<fam>:<other>/<id>" (verifier: 0)
Each trace is re-notated under a cycled delimiter scheme (label-invariant,
re-verified), so selection also reflects notation robustness.

Usage:
    PYTHONPATH=. python3 make_transfer_val.py --seed 303 --n-namespaces 40
    # writes transfer_val.jsonl and prints the verifier's label split
"""

from __future__ import annotations

import argparse
import math
import random

from authority_verifier import Delegation, Grant, RootAuthority, Scope
from augment_representation import augment
from map_tau_to_chain import _action_record, label_split
from trace_benchmark import delegation_to_json, scope_to_json, write_jsonl

# synthetic vocabulary only — deliberately disjoint from tau2/Toucan tool names
FAMILY = "acct"
TOOLS = ("fetch_record", "update_record", "close_case", "send_notice",
         "adjust_plan", "issue_credit", "list_devices", "reset_access")


def build_traces(seed: int, n_namespaces: int, calls_per_ns: int) -> list:
    rng = random.Random(seed)
    root = RootAuthority("core_system",
                         Scope((Grant("*", f"{FAMILY}:*", math.inf),)))
    traces = []
    for ns in range(n_namespaces):
        tools = sorted(rng.sample(TOOLS, k=rng.randint(2, 4)))
        grants = tuple(Grant(t, f"{FAMILY}:{ns}/*", math.inf) for t in tools)
        hop = Delegation("core_system", "task_agent", Scope(grants), issued_at=0)
        chain = [hop]
        for j in range(calls_per_ns):
            tool = rng.choice(tools)
            rid = f"rec-{rng.randint(100, 999)}"
            other = rng.choice([x for x in range(n_namespaces) if x != ns])
            auth = _action_record("task_agent", tool,
                                  f"{FAMILY}:{ns}/{rid}", 0.0, root, chain)
            redir = _action_record("task_agent", tool,
                                   f"{FAMILY}:{other}/rec-{rng.randint(100, 999)}",
                                   0.0, root, chain)
            traces.append({
                "trace_id": f"tval-{ns:03d}-{j:02d}",
                "scenario_class": "attack_confused_deputy",
                "note": "synthetic transfer-val: real-mapping structure, "
                        "synthetic vocabulary",
                "root": {"principal": root.principal,
                         "scope": scope_to_json(root.scope)},
                "delegations": [delegation_to_json(hop)],
                "actions": [auth, redir],
            })
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=303,
                    help="disjoint from every training/val/test seed in use")
    ap.add_argument("--n-namespaces", type=int, default=40)
    ap.add_argument("--calls-per-ns", type=int, default=2)
    ap.add_argument("--out", default="transfer_val.jsonl")
    args = ap.parse_args()

    traces = build_traces(args.seed, args.n_namespaces, args.calls_per_ns)
    # mixed notations, labels re-verified (0 discards expected)
    mixed, discarded = augment(traces, seed=args.seed + 1)
    write_jsonl(mixed, args.out)
    from collections import Counter
    dist = Counter(t["_notation"] for t in mixed)
    print(f"{args.out}: {len(mixed)} traces, split {label_split(mixed)}, "
          f"notations {dict(dist)}, discarded {discarded}")


if __name__ == "__main__":
    main()
