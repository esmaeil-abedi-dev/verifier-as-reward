"""
map_tau_to_chain.py
==================

E5 — real-trace external validity. Maps independently-authored tau2-bench
agent trajectories (Jarrodbarnes/tau2-sft-v4-dataset, Apache-2.0) into our
(root, delegation chain, action) schema so that OUR verifier assigns every
label. No hand-labeling; the verifier is the sole source of truth, exactly
as everywhere else in the paper.

Mapping (single principal -> single-hop chain):
  - Each tau2 trajectory serves ONE customer. The support *system* holds
    authority over every customer (`cust:*`); it delegates to the agent
    authority over only the served customer (`cust:<tid>:*`). One hop.
  - The agent's real tool calls (parsed from the assistant turns' JSON) that
    reference the served customer's resource IDs become AUTHORIZED actions:
    action = tool name, resource = `cust:<tid>:<resource-id>`.
  - Each such call is also emitted REDIRECTED to a *different* trajectory's
    real resource id (`cust:<other>:<foreign-id>`) — a confused deputy on a
    real call: the system (root) could act on it, the agent (its narrowed
    scope) may not. The verifier labels these unauthorized.

Every action's label comes from `label_action`. We do NOT force a 50/50
split: the verifier adjudicates each action and we report the actual
authorized/unauthorized counts.

Honest limits (state in the paper): single-hop (single principal), so
attenuated-delegation structure is synthetic in the mapping; the redirected
unauthorized cases are scope-violations *constructed* on real calls (real
tool names, real args, real foreign ids — but the redirect is a
perturbation, not a naturally-occurring attack). AgentDojo's native-injection
evaluation is a different (prompt-injection) threat model, cited as
complementary future work rather than run.

Usage:
    PYTHONPATH=. python3 map_tau_to_chain.py --per-class 0 --seed 5
    # writes real_trace_<domain>.jsonl for telecom/airline/retail + a combined
    # real_trace_all.jsonl, and prints the mapping-fidelity / label split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re

from authority_verifier import (
    Action, Delegation, Grant, RootAuthority, Scope, label_action, verify,
)
from trace_benchmark import delegation_to_json, scope_to_json

DATASET = "Jarrodbarnes/tau2-sft-v4-dataset"
DATASET_FILE = "blended_traces_v4.jsonl"

def _is_id_key(k: str) -> bool:
    """A key naming a single resource id (order_id, line_id, reservation_id)."""
    return k == "id" or k.endswith("_id")


def _is_id_collection_key(k: str) -> bool:
    """A key naming one or many ids (adds plural line_ids, bill_ids, ...)."""
    return _is_id_key(k) or k.endswith("_ids")


def load_tau(limit: int = 0) -> list:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit else rows


def domain_of(row: dict) -> str:
    # task_id looks like "[telecom][scenario]..."
    m = re.match(r"\[([a-z]+)\]", row["task_id"])
    return m.group(1) if m else "unknown"


def parse_tool_call(assistant_content: str):
    """Return (name, arguments) for an assistant turn that issues a tool call,
    else None. The call is a JSON object after any <thinking>...</thinking>."""
    if not assistant_content:
        return None
    text = re.sub(r"<thinking>.*?</thinking>", "", assistant_content,
                  flags=re.DOTALL).strip()
    # parse the FIRST complete JSON object (rfind would grab the inner
    # "arguments": {...} brace, not the outer {"name": ..., "arguments": ...})
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj:
        return obj["name"], obj.get("arguments", {}) or {}
    return None


def ids_in(obj) -> list:
    """All string values under an id-like key (singular or plural), anywhere
    in a nested structure — domain-agnostic (C1001, L1001, #W2378156,
    EHGLP3, ...)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_id_collection_key(k):
                if isinstance(v, str):
                    out.append(v)
                elif isinstance(v, list):
                    out += [x for x in v if isinstance(x, str)]
            out += ids_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += ids_in(v)
    return out


def call_resource_id(arguments: dict):
    """The resource id a tool call acts on: the first argument value under an
    id-like key (any value format — retail '#W...', airline 'EHGLP3', telecom
    'L1001')."""
    for k, v in arguments.items():
        if _is_id_key(k) and isinstance(v, str) and v:
            return v
    return None


def call_amount(arguments: dict) -> float:
    """A numeric amount in the call, if any (for the optional budget variant)."""
    for k, v in arguments.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if any(t in k.lower() for t in ("amount", "credit", "price",
                                            "cost", "total", "fee")):
                return float(v)
    return 0.0


def trajectory_calls(row: dict) -> list:
    """(tool_name, resource_id, amount) for each assistant tool call that
    targets a resource id."""
    calls = []
    for m in row["prompt"]:
        if m.get("role") != "assistant":
            continue
        pc = parse_tool_call(m.get("content", ""))
        if not pc:
            continue
        name, args = pc
        rid = call_resource_id(args)
        if rid is not None:
            calls.append((name, rid, call_amount(args)))
    return calls


def all_trajectory_ids(row: dict) -> set:
    """Every id appearing anywhere in a trajectory (args + tool results) —
    the served customer's resource namespace."""
    ids = set()
    for m in row["prompt"]:
        c = m.get("content", "")
        if isinstance(c, str) and c.strip().startswith("{"):
            try:
                ids |= set(ids_in(json.loads(c)))
            except json.JSONDecodeError:
                pass
        elif m.get("role") == "assistant":
            pc = parse_tool_call(c)
            if pc:
                ids |= set(ids_in(pc[1]))
    return ids


def _root_and_hop(tid: int):
    """System holds all-customer authority; delegates the served customer."""
    root = RootAuthority("support_system",
                         Scope((Grant("*", "cust:*", math.inf),)))
    hop = Delegation("support_system", "support_agent",
                     Scope((Grant("*", f"cust:{tid}:*", math.inf),)),
                     issued_at=0)
    return root, hop


def _action_record(agent, tool, resource, amount, root, chain):
    act = Action(agent, tool, resource, amount, t=1)
    v = verify(act, chain, root)
    return {
        "agent": agent, "action": tool, "resource": resource,
        "amount": amount, "t": 1,
        "label": 1 if v.authorized else 0,
        "failing_hop": v.failing_hop, "reason": v.reason,
    }


def build_traces(rows: list, seed: int):
    """One trace per (trajectory, call): the authorized in-scope action and a
    redirected out-of-scope action, both verifier-labeled. Returns
    (traces, stats)."""
    rng = random.Random(seed)
    # pool of (tid, id) for redirects
    per_traj_ids = [(i, sorted(all_trajectory_ids(r))) for i, r in enumerate(rows)]
    traces = []
    n_calls = 0
    for i, row in enumerate(rows):
        tid = i
        dom = domain_of(row)
        root, hop = _root_and_hop(tid)
        chain = [hop]
        for j, (tool, rid, amount) in enumerate(trajectory_calls(row)):
            n_calls += 1
            # a foreign (tid, id) from a DIFFERENT trajectory
            others = [(t, ids) for t, ids in per_traj_ids if t != tid and ids]
            ft, fids = rng.choice(others)
            foreign_id = rng.choice(fids)
            auth = _action_record(
                "support_agent", tool, f"cust:{tid}:{rid}", amount, root, chain)
            redir = _action_record(
                "support_agent", tool, f"cust:{ft}:{foreign_id}", amount, root, chain)
            traces.append({
                "trace_id": f"tau-{dom}-{tid:04d}-{j:02d}",
                "scenario_class": "attack_confused_deputy",
                "note": (f"real tau2 {dom} call '{tool}' on served customer "
                         f"{rid}; redirect targets foreign {ft}:{foreign_id}"),
                "root": {"principal": root.principal,
                         "scope": scope_to_json(root.scope)},
                "delegations": [delegation_to_json(hop)],
                "actions": [auth, redir],
            })
    stats = {"n_trajectories": len(rows), "n_calls_extracted": n_calls,
             "n_traces": len(traces)}
    return traces, stats


def label_split(traces: list) -> dict:
    labels = [a["label"] for t in traces for a in t["actions"]]
    return {"n_actions": len(labels), "authorized": sum(labels),
            "unauthorized": len(labels) - sum(labels)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap trajectories (0 = all)")
    ap.add_argument("--out-prefix", default="real_trace")
    args = ap.parse_args()

    rows = load_tau(args.limit)
    from collections import Counter
    doms = Counter(domain_of(r) for r in rows)
    print(f"loaded {len(rows)} tau2 trajectories: {dict(doms)}")

    traces, stats = build_traces(rows, args.seed)
    print(f"extracted {stats['n_calls_extracted']} resource-targeting calls "
          f"-> {stats['n_traces']} traces")

    # write per-domain + combined
    from trace_benchmark import write_jsonl
    by_dom = {}
    for tr in traces:
        by_dom.setdefault(tr["trace_id"].split("-")[1], []).append(tr)
    for dom, trs in sorted(by_dom.items()):
        write_jsonl(trs, f"{args.out_prefix}_{dom}.jsonl")
        print(f"  {dom}: {label_split(trs)}")
    write_jsonl(traces, f"{args.out_prefix}_all.jsonl")
    print(f"combined {args.out_prefix}_all.jsonl: {label_split(traces)}")


if __name__ == "__main__":
    main()
