"""
dump_predictions.py
==================

Write per-action prediction records for a backend (E2, qualitative analysis).
Unlike `eval_harness.py`, which persists only aggregate metrics by default,
this dumps one row per action:

    {trace_id, scenario_class, chain_len, has_decoy, tight_window,
     prompt, label, prediction, failing_hop_true, model_reply_raw}

so error modes can be inspected offline (which traces each model gets right
or wrong, and how misclassifications distribute by scenario class and by
structural feature). Labels are the verifier's, carried in the corpus and
re-derivable from it; nothing here is hand-labeled.

Backends: a baseline name (`heuristic`/`random`/`always_authorized`/`oracle`),
`openrouter:<model-id>`, or `local:<checkpoint-dir-or-hub-id>`.

Usage:
    PYTHONPATH=. python3 dump_predictions.py \
        --backend local:esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b \
        --test-file benchmark_test.jsonl \
        --out predictions_ce.jsonl
    PYTHONPATH=. python3 dump_predictions.py \
        --backend openrouter:anthropic/claude-sonnet-4.5 --out predictions_sonnet.jsonl
"""

from __future__ import annotations

import argparse
import json

from eval_harness import (
    load_dotenv,
    make_backends,
    make_openrouter_backend,
    run_eval,
)
from trace_benchmark import load_traces


def _structural_features(traces: list) -> dict:
    """Per-(trace_id, t) structural descriptors used to slice error modes."""
    feats = {}
    for tr in traces:
        chain_len = len(tr["delegations"])
        has_decoy = any(len(d["scope"]["grants"]) > 1 for d in tr["delegations"])
        for aj in tr["actions"]:
            windows = [w for d in tr["delegations"]
                       for w in (d["revoked_at"], d["expires_at"])
                       if w is not None]
            tight = any(0 <= (w - aj["t"]) <= 3 for w in windows)
            feats[(tr["trace_id"], aj["t"])] = {
                "chain_len": chain_len,
                "has_decoy": has_decoy,
                "tight_window": tight,
            }
    return feats


def resolve_backend(name: str, traces: list, seed: int):
    """A baseline name, `openrouter:<id>`, or `local:<path>`."""
    if name.startswith("openrouter:"):
        return make_openrouter_backend(name.split(":", 1)[1])
    if name.startswith("local:"):
        # imported lazily so the offline baselines/openrouter path needs no torch
        from train_verifier_reward import make_checkpoint_backend, pick_device
        return make_checkpoint_backend(name.split(":", 1)[1], pick_device("auto"))
    backends = make_backends(traces, seed)
    if name in backends:
        return backends[name]
    raise SystemExit(f"unknown backend {name!r}; use a baseline "
                     f"{sorted(backends)}, openrouter:<id>, or local:<path>")


def dump(backend_name: str, test_file: str, out_path: str, seed: int) -> dict:
    load_dotenv()
    traces = load_traces(test_file)
    feats = _structural_features(traces)
    answer_fn = resolve_backend(backend_name, traces, seed)
    out = run_eval(answer_fn, traces)
    with open(out_path, "w") as f:
        for r in out["records"]:
            fe = feats.get((r["trace_id"], r["t"]), {})
            f.write(json.dumps({
                "backend": backend_name,
                "trace_id": r["trace_id"],
                "scenario_class": r["scenario_class"],
                "chain_len": fe.get("chain_len"),
                "has_decoy": fe.get("has_decoy"),
                "tight_window": fe.get("tight_window"),
                "prompt": r["prompt"],
                "label": r["label"],
                "prediction": r["prediction"],
                "failing_hop_true": r["failing_hop_true"],
                "model_reply_raw": r["raw_reply"],
                "error": r["error"],
            }) + "\n")
    m = out["metrics"]
    print(f"{backend_name}: {len(out['records'])} records -> {out_path}  "
          f"(acc {m['accuracy']:.3f}, false-authorize "
          f"{m['headline_false_authorize_rate_on_violation_classes']})")
    return out["metrics"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--backend", required=True)
    ap.add_argument("--test-file", default="benchmark_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dump(args.backend, args.test_file, args.out, args.seed)


if __name__ == "__main__":
    main()
