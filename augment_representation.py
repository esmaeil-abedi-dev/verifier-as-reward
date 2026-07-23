"""
augment_representation.py
========================

Representation-augmentation for the training corpus, to make the trained model
robust to how resources are *notated* (the brittleness E5 surfaced: the model
enforces authorization on the trained `family:namespace/leaf` notation but
over-authorizes when the same structure is rendered with different delimiters).

The synthetic benchmark renders resources as `family:namespace/leaf` (colon
between family and namespace, slash before the leaf), e.g. `inbox:alice/msg-1`
with scope glob `inbox:alice/*`. This script emits, per trace, a resource
notation drawn from several delimiter schemes, applied **consistently** to
every resource string in the trace (root scope, every hop's scope, and every
action). Because `fnmatch` and the verifier's subsumption logic special-case
only glob metacharacters (`*?[]`) and never the delimiters `:`/`/`, a
consistent delimiter substitution **preserves every verdict** — which we
re-check with the verifier and assert (discarding, and counting, any trace
whose label changes; expected 0).

Training CE on the mixed-notation corpus teaches resource-scope enforcement
that does not depend on the surface delimiter, so real-data mappings in any of
these notations are handled.

Schemes (delimiters that never collide with ids `msg-1`/`#W..` or action dots):
  canonical  family:namespace/leaf     (unchanged; the trained default)
  allcolon   family:namespace:leaf     (the naive tau2 mapping that scored 55%)
  allslash   family/namespace/leaf
  pipe       family|namespace|leaf

Usage:
    PYTHONPATH=. python3 augment_representation.py \
        --in expanded_train.jsonl --out augmented_train.jsonl --seed 33
"""

from __future__ import annotations

import argparse
import copy
import json
import random

from authority_verifier import label_action
from trace_benchmark import load_traces, trace_to_objects, write_jsonl

# Each scheme maps the canonical delimiters (':' family|ns, '/' ns|leaf) to a
# (family_sep, leaf_sep) pair. Canonical is the identity.
SCHEMES = {
    "canonical": (":", "/"),
    "allcolon":  (":", ":"),
    "allslash":  ("/", "/"),
    "pipe":      ("|", "|"),
}


def renotate_resource(res: str, scheme: str) -> str:
    """Re-render a resource/glob string in `scheme`. Canonical resources use
    ':' before the namespace and '/' before the leaf; we replace those with
    the scheme's separators, leaving glob '*' and the id text untouched."""
    fam_sep, leaf_sep = SCHEMES[scheme]
    return res.replace("/", leaf_sep).replace(":", fam_sep) \
        if scheme != "canonical" else res


def _renotate_scope(scope: dict, scheme: str) -> dict:
    return {"grants": [{**g, "resource": renotate_resource(g["resource"], scheme)}
                       for g in scope["grants"]]}


def renotate_trace(trace: dict, scheme: str) -> dict:
    """A copy of `trace` with every resource string re-notated in `scheme`."""
    t = copy.deepcopy(trace)
    t["root"]["scope"] = _renotate_scope(t["root"]["scope"], scheme)
    for d in t["delegations"]:
        d["scope"] = _renotate_scope(d["scope"], scheme)
    for a in t["actions"]:
        a["resource"] = renotate_resource(a["resource"], scheme)
    t["_notation"] = scheme
    return t


def label_preserved(trace: dict) -> bool:
    """Every action's stored label equals a fresh verifier verdict on the
    re-notated (root, chain, action)."""
    root, chain, actions = trace_to_objects(trace)
    return all(label_action(a, chain, root) == aj["label"]
               for a, aj in zip(actions, trace["actions"]))


def augment(traces: list, seed: int, schemes: list = None):
    """Each trace re-notated under a randomly chosen scheme (schemes cycled so
    the mix is balanced). Returns (augmented_traces, n_discarded)."""
    schemes = schemes or list(SCHEMES)
    rng = random.Random(seed)
    order = list(traces)
    rng.shuffle(order)
    out, discarded = [], 0
    for i, tr in enumerate(order):
        scheme = schemes[i % len(schemes)]
        aug = renotate_trace(tr, scheme)
        if label_preserved(aug):      # guard: notation must not change verdicts
            out.append(aug)
        else:
            discarded += 1
    out.sort(key=lambda t: t["trace_id"])
    return out, discarded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in", dest="in_file", default="expanded_train.jsonl")
    ap.add_argument("--out", default="augmented_train.jsonl")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--schemes", default=",".join(SCHEMES),
                    help="comma-separated notation schemes to mix")
    args = ap.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    unknown = set(schemes) - set(SCHEMES)
    if unknown:
        raise SystemExit(f"unknown schemes {unknown}; available {list(SCHEMES)}")

    traces = load_traces(args.in_file)
    aug, discarded = augment(traces, args.seed, schemes)
    write_jsonl(aug, args.out)
    from collections import Counter
    dist = Counter(t["_notation"] for t in aug)
    print(f"{args.out}: {len(aug)} traces "
          f"({sum(len(t['actions']) for t in aug)} actions); "
          f"{discarded} discarded (label changed under re-notation). "
          f"notation mix: {dict(dist)}")


if __name__ == "__main__":
    main()
