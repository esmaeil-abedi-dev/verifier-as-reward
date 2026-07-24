"""
map_toucan_to_chain.py
=====================

E5 (second real source) — external validity from an INDEPENDENT real corpus.
Maps Toucan-1.5M agent trajectories (Agent-Ark/Toucan-1.5M, Apache-2.0 — real
multi-server MCP tool-use, authored with no knowledge of our verifier) into our
(root, delegation chain, action) schema so OUR verifier assigns every label.
No hand-labeling; the verifier is the sole source of truth, as everywhere else.

Why a second source. tau2 (`map_tau_to_chain.py`) is three customer-support
domains with `_id` arguments. Toucan is thousands of *different* MCP servers
(web search, Unity, trivia, finance, ...) with heterogeneous tool vocabularies
and argument shapes. If the model recognizes real authorized calls here too, the
"transfers to real tool-call vocabulary" claim is not a tau2 artifact.

Mapping (single principal -> single-hop chain, mirrors the tau2 mapping):
  - Each trajectory is one *session*. A system holds authority over every
    session (`sess:*`) and delegates to the agent authority over only this
    session (`sess:<sid>/*`), granting the SPECIFIC tools the agent uses (not a
    wildcard action — the trained grants always name a concrete action).
  - Each real tool call (assistant turn's `function_call`) becomes an
    AUTHORIZED action: action = tool name, resource = `sess:<sid>/<leaf>`,
    where <leaf> is a slug of the call's primary argument (real-derived).
  - Optionally (`--redirect`) each call is also emitted REDIRECTED to a foreign
    session's real leaf — a confused deputy on a real call (verifier-unauth),
    giving a balanced set. Default is authorized-only: Toucan is good-behaviour
    data, so the headline use is a *recognition* test (false-refuse rate on real
    legitimate calls). The tau2 balanced set carries the false-authorize test.

Every label comes from `label_action`; we report the actual authorized/
unauthorized counts, never forcing a split.

Honest limits (same as tau2): single principal, so the attenuated-delegation
structure is synthetic in the mapping; the redirect (if enabled) is a
constructed scope-violation on real calls, not a naturally-occurring attack.

Usage:
    PYTHONPATH=. python3 map_toucan_to_chain.py --shards 1 --max-traj 150 --seed 5
    # writes real_toucan_all.jsonl (+ per-server splits) and prints the split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re

from authority_verifier import Delegation, Grant, RootAuthority, Scope
from trace_benchmark import delegation_to_json, scope_to_json, write_jsonl
# generic verifier-labeling helpers, shared with the tau2 mapper
from map_tau_to_chain import _action_record, label_split

DATASET = "Agent-Ark/Toucan-1.5M"
SUBSET = "Kimi-K2"  # the released split directory of parquet shards

# argument keys that most directly name the resource a call acts on, in
# rough priority order (checked case-insensitively, substring match)
_ID_KEY_HINTS = ("_id", "id", "task", "path", "file", "url", "uri", "symbol",
                 "ticker", "query", "name", "key", "slug", "channel", "repo")


def _slug(value, fallback: str) -> str:
    """A short, delimiter-free resource leaf derived from a real argument value.
    Keeps [a-z0-9-] only (so no ':'/'/' survives to break renotation), lowercased,
    truncated. Missing/empty -> fallback (so every call yields a distinct leaf)."""
    if value is None:
        return fallback
    s = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    s = s[:24].strip("-")
    return s or fallback


def _primary_arg(arguments: dict):
    """The value most likely to identify the resource: first value whose key
    matches an id-hint, else the first scalar argument value."""
    if not isinstance(arguments, dict):
        return None
    for k, v in arguments.items():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool) \
                and any(h in k.lower() for h in _ID_KEY_HINTS):
            return v
    for v in arguments.values():          # fallback: any scalar
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            return v
    return None


def _clean_tool(name: str) -> str:
    """A tool/action name safe as a verifier action (no ':'/'/' delimiters)."""
    return re.sub(r"[:/]", "-", str(name)) if name else name


def trajectory_calls(messages: list) -> list:
    """(tool_name, resource_leaf) for each assistant `function_call` that names a
    tool. Toucan puts the call in an assistant turn's `function_call` key; its
    `arguments` is a JSON string."""
    calls = []
    for j, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        fc = m.get("function_call")
        if not fc or not fc.get("name"):
            continue
        args = fc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        leaf = _slug(_primary_arg(args), fallback=f"call{j}")
        calls.append((_clean_tool(fc["name"]), leaf))
    return calls


def load_toucan(shards: int = 1, max_traj: int = 0) -> list:
    """Return [(session_id, [messages]), ...] for trajectories with >=1 tool
    call, reading the first `shards` parquet shards of the Kimi-K2 split."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(DATASET, repo_type="dataset")
                   if f.startswith(f"{SUBSET}/") and f.endswith(".parquet"))
    out = []
    for fn in files[:max(1, shards)]:
        path = hf_hub_download(DATASET, fn, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for r in batch.to_pylist():
                try:
                    msgs = json.loads(r["messages"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if any(m.get("role") == "assistant" and m.get("function_call")
                       for m in msgs):
                    out.append((r.get("uuid", f"t{len(out)}"), msgs))
                    if max_traj and len(out) >= max_traj:
                        return out
    return out


def _root_and_hop(sid: int, tools: list):
    """System holds all-session authority; delegates this session's namespace,
    granting each SPECIFIC tool the agent uses. Resources use the trained
    `family:namespace/leaf` hierarchy (`sess:<sid>/<leaf>`)."""
    root = RootAuthority("mcp_system", Scope((Grant("*", "sess:*", math.inf),)))
    grants = tuple(Grant(t, f"sess:{sid}/*", math.inf) for t in tools)
    hop = Delegation("mcp_system", "mcp_agent", Scope(grants), issued_at=0)
    return root, hop


def build_traces(trajs: list, seed: int, redirect: bool = False):
    """One trace per (session, call). Authorized in-scope action always; a
    redirected out-of-scope action too when `redirect`. Returns (traces, stats)."""
    rng = random.Random(seed)
    parsed = [(sid, trajectory_calls(msgs)) for sid, msgs in trajs]
    parsed = [(i, sid, calls) for i, (sid, calls) in enumerate(parsed) if calls]
    # redirect pool: (session_index, [leaves]) for foreign targets
    leaves_by_traj = [(i, sorted({leaf for _, leaf in calls}))
                      for i, _, calls in parsed]
    traces, n_calls = [], 0
    for i, sid, calls in parsed:
        tools = sorted({tool for tool, _ in calls})
        root, hop = _root_and_hop(i, tools)
        chain = [hop]
        server = calls[0][0].split("-")[0] or "mcp"   # server slug for splits
        for j, (tool, leaf) in enumerate(calls):
            n_calls += 1
            actions = [_action_record(
                "mcp_agent", tool, f"sess:{i}/{leaf}", 0.0, root, chain)]
            if redirect:
                others = [(t, ls) for t, ls in leaves_by_traj if t != i and ls]
                if others:
                    ft, fls = rng.choice(others)
                    actions.append(_action_record(
                        "mcp_agent", tool, f"sess:{ft}/{rng.choice(fls)}",
                        0.0, root, chain))
            traces.append({
                "trace_id": f"toucan-{server}-{i:04d}-{j:02d}",
                "scenario_class": "attack_confused_deputy" if redirect
                                  else "single_delegation",
                "note": (f"real Toucan call '{tool}' on session {sid} leaf "
                         f"'{leaf}'" + (" (+redirect)" if redirect else "")),
                "root": {"principal": root.principal,
                         "scope": scope_to_json(root.scope)},
                "delegations": [delegation_to_json(hop)],
                "actions": actions,
            })
    stats = {"n_trajectories": len(parsed), "n_calls_extracted": n_calls,
             "n_traces": len(traces)}
    return traces, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--shards", type=int, default=1, help="parquet shards to read")
    ap.add_argument("--max-traj", type=int, default=150,
                    help="cap trajectories with >=1 tool call (0 = all in shards)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--redirect", action="store_true",
                    help="also emit a foreign-session redirect per call (balanced)")
    ap.add_argument("--out-prefix", default="real_toucan")
    args = ap.parse_args()

    trajs = load_toucan(args.shards, args.max_traj)
    print(f"loaded {len(trajs)} Toucan trajectories with tool calls "
          f"(from {args.shards} shard(s))")

    traces, stats = build_traces(trajs, args.seed, redirect=args.redirect)
    print(f"extracted {stats['n_calls_extracted']} tool calls "
          f"-> {stats['n_traces']} traces")

    by_server = {}
    for tr in traces:
        by_server.setdefault(tr["trace_id"].split("-")[1], []).append(tr)
    # write only the combined file + a compact per-server summary (many servers)
    write_jsonl(traces, f"{args.out_prefix}_all.jsonl")
    print(f"combined {args.out_prefix}_all.jsonl: {label_split(traces)}")
    print(f"servers: {len(by_server)} distinct "
          f"(e.g. {sorted(by_server)[:6]})")


if __name__ == "__main__":
    main()
