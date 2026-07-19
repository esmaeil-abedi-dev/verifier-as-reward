"""
train_verifier_reward.py
========================

Verifier-as-reward training harness.

Trains a causal LM policy to decide AUTHORIZE / REFUSE for benchmark actions,
with the deterministic verifier's verdict (`label_action`) as the reward:

    reward = +1  if the policy's decision matches the verifier
             -1  if it false-authorizes or false-refuses

The policy scores two candidate continuations (" AUTHORIZE" / " REFUSE") of a
compact trace prompt by their length-normalized (mean per-token) sequence
log-probability — normalization keeps the unequal candidate lengths from
biasing the initial policy to one decision — samples a decision from the
softmax over the two scores, and is updated by REINFORCE with a running-mean
baseline.
The loop is modular: `load_examples`, `reward_for_decision`, and
`candidate_logprobs` are exactly the pieces a TRL/PPO trainer would reuse —
swap the update rule without touching the reward path.

Two model backends:

  - `--model tiny` (default): a randomly initialized 2-layer GPT-2 over raw
    bytes, built locally with no downloads and no network. This is the CPU
    smoke test: it proves the full loop (prompting, sampling, verifier
    reward, policy gradient, held-out eval, logging) end to end in seconds.
  - `--model <hf-name>` (e.g. gpt2, Qwen/Qwen2.5-0.5B): any Hugging Face
    causal LM, for real runs on a GPU.

If no CUDA GPU is present, the harness runs whatever you asked for (the
smoke test by default) and prints a clear "ready for GPU" message with the
scale-up command.

Logging: every eval writes a JSON line to training_log.jsonl with the step,
mean training reward, and — on a seeded, shuffled held-out subset —
accuracy, violation rate (false-authorize rate on unauthorized actions),
and false-refuse rate, versus cumulative training examples. The
does-it-improve-with-scale claim must rest on accuracy (or on violation
rate AND false-refuse rate jointly): violation rate alone collapses to 0
for a policy that degenerates to always-refuse, which is failure, not
learning.

Usage (CPU smoke test):
    PYTHONPATH=. python3 train_verifier_reward.py

Scale up (GPU):
    PYTHONPATH=. python3 train_verifier_reward.py \
        --model gpt2 --steps 500 --batch-size 16 --lr 1e-5 --eval-every 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch

from authority_verifier import label_action
from trace_benchmark import load_traces, trace_to_objects

DECISIONS = (" AUTHORIZE", " REFUSE")  # index 0 -> authorize (decision=1)


# --------------------------------------------------------------------------
# Byte-level tokenizer (no downloads; used by the tiny smoke-test model)
# --------------------------------------------------------------------------

class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8", errors="replace"))


def build_model_and_tokenizer(name: str, seed: int):
    """`tiny` builds a local random-weight model (offline). Anything else is
    resolved through Hugging Face (may download on first use)."""
    torch.manual_seed(seed)
    if name == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=256, n_positions=1024,
                            n_embd=32, n_layer=2, n_head=2)
        return GPT2LMHeadModel(config), ByteTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)

    class _HFWrapper:
        hf_tokenizer = tok  # kept for checkpoint saving

        def encode(self, text: str) -> list:
            return tok.encode(text, add_special_tokens=False)

    return model, _HFWrapper()


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Examples and reward
# --------------------------------------------------------------------------

def compact_prompt(trace: dict, action: dict) -> str:
    """A terse, byte-cheap rendering of the chain + pending action (the
    natural-language rendering in eval_harness is for frontier models; this
    one keeps tiny-model contexts short)."""
    def scope_s(scope):
        return ",".join(
            f"{g['action']}|{g['resource']}|"
            f"{'inf' if g['max_budget'] is None else g['max_budget']}"
            for g in scope["grants"])

    lines = [f"ROOT {trace['root']['principal']} "
             f"[{scope_s(trace['root']['scope'])}]"]
    for i, d in enumerate(trace["delegations"]):
        exp = "inf" if d["expires_at"] is None else d["expires_at"]
        rev = "-" if d["revoked_at"] is None else d["revoked_at"]
        lines.append(f"HOP{i} {d['delegator']}>{d['delegatee']} "
                     f"[{scope_s(d['scope'])}] "
                     f"t{d['issued_at']}:{exp} rev={rev}")
    a = action
    lines.append(f"ACT {a['agent']} {a['action']} {a['resource']} "
                 f"amt={a['amount']} t={a['t']}")
    lines.append("DECISION:")
    return "\n".join(lines)


def load_examples(path: str) -> list:
    """One example per action: the prompt plus the verifier objects needed to
    compute the reward at decision time."""
    examples = []
    for tr in load_traces(path):
        root, chain, actions = trace_to_objects(tr)
        for act, aj in zip(actions, tr["actions"]):
            examples.append({
                "prompt": compact_prompt(tr, aj),
                "action": act,
                "chain": chain,
                "root": root,
                "scenario_class": tr["scenario_class"],
            })
    return examples


def reward_for_decision(decision: int, example: dict) -> float:
    """+1 iff the decision matches the verifier's verdict, else -1. The
    verifier is the sole reward source — labels stored in the corpus are
    never read here."""
    verdict = label_action(example["action"], example["chain"], example["root"])
    return 1.0 if decision == verdict else -1.0


# --------------------------------------------------------------------------
# Policy: score the two candidate continuations
# --------------------------------------------------------------------------

def candidate_logprobs(model, tokenizer, prompt: str,
                       device: torch.device) -> torch.Tensor:
    """Length-normalized (mean per-token) log-probability of each decision
    continuation given the prompt. Normalization matters: the candidates
    tokenize to different lengths, and raw sequence log-probs would bias an
    untrained policy almost entirely toward the shorter one. Returns a
    tensor of shape (2,) with gradients attached."""
    prompt_ids = tokenizer.encode(prompt)
    lps = []
    for cand in DECISIONS:
        cand_ids = tokenizer.encode(cand)
        ids = torch.tensor([prompt_ids + cand_ids], device=device)
        logits = model(input_ids=ids).logits[0]
        start = len(prompt_ids) - 1
        # slice to the candidate positions BEFORE the fp32 softmax: a full
        # sequence x vocab log-softmax kept in the graph costs hundreds of
        # MB per forward for real vocabularies (Qwen: ~152k)
        sel = logits[start:start + len(cand_ids)].float()
        logprobs = torch.log_softmax(sel, dim=-1)
        lp = sum(logprobs[j, cand_ids[j]] for j in range(len(cand_ids)))
        lps.append(lp / len(cand_ids))
    return torch.stack(lps)


@torch.no_grad()
def evaluate(model, tokenizer, examples: list, device: torch.device) -> dict:
    """Greedy-decision metrics on held-out examples. The violation rate is
    the false-authorize rate on actions the verifier rejects; the
    false-refuse rate is its mirror on actions the verifier authorizes.
    Report them together — either alone can be driven to 0 by a degenerate
    always-refuse / always-authorize policy."""
    if not examples:
        return {"n_eval_actions": 0, "accuracy": None,
                "heldout_violation_rate": None,
                "heldout_false_refuse_rate": None}
    model.eval()
    n_correct = 0
    n_viol = n_false_auth = 0
    n_auth = n_false_refuse = 0
    for ex in examples:
        lps = candidate_logprobs(model, tokenizer, ex["prompt"], device)
        decision = 1 if int(torch.argmax(lps)) == 0 else 0
        verdict = label_action(ex["action"], ex["chain"], ex["root"])
        n_correct += int(decision == verdict)
        if verdict == 0:
            n_viol += 1
            n_false_auth += int(decision == 1)
        else:
            n_auth += 1
            n_false_refuse += int(decision == 0)
    model.train()
    return {
        "n_eval_actions": len(examples),
        "accuracy": n_correct / len(examples),
        "heldout_violation_rate":
            (n_false_auth / n_viol) if n_viol else None,
        "heldout_false_refuse_rate":
            (n_false_refuse / n_auth) if n_auth else None,
    }


# --------------------------------------------------------------------------
# REINFORCE loop
# --------------------------------------------------------------------------

def train(args) -> list:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    if args.save_dir:  # fail on an unusable path now, not after N steps
        os.makedirs(args.save_dir, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(args.model, args.seed)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_examples = load_examples(args.train_file)
    # Seeded shuffle before truncating: a plain [:n] slice of the trace-id-
    # sorted list would silently drop whole scenario classes from the eval.
    eval_examples = load_examples(args.test_file)
    random.Random(args.seed).shuffle(eval_examples)
    eval_examples = eval_examples[: args.eval_max_actions]
    print(f"device={device.type} model={args.model} "
          f"train_actions={len(train_examples)} eval_actions={len(eval_examples)}")

    history = []
    baseline = 0.0
    log_f = open(args.log_file, "w")
    # First log line records the exact command + config for reproducibility.
    log_f.write(json.dumps({"type": "config", "argv": sys.argv,
                            "args": vars(args), "device": device.type}) + "\n")
    log_f.flush()

    def log_point(step: int, mean_reward, loss):
        point = {
            "step": step,
            "cumulative_examples": step * args.batch_size,
            "mean_reward": mean_reward,
            "loss": loss,
            **evaluate(model, tokenizer, eval_examples, device),
        }
        history.append(point)
        log_f.write(json.dumps(point) + "\n")
        log_f.flush()
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"
        print(f"step {step:>4}  reward {str(mean_reward):>6}  "
              f"heldout acc {fmt(point['accuracy'])}  "
              f"violation rate {fmt(point['heldout_violation_rate'])}  "
              f"false-refuse {fmt(point['heldout_false_refuse_rate'])}")

    log_point(0, None, None)  # untrained baseline point of the curve
    for step in range(1, args.steps + 1):
        batch = [rng.choice(train_examples) for _ in range(args.batch_size)]
        rewards = []
        step_loss = torch.zeros((), device=device)  # one host sync per step
        opt.zero_grad()
        for ex in batch:
            lps = candidate_logprobs(model, tokenizer, ex["prompt"], device)
            probs = torch.softmax(lps.detach(), dim=0)
            idx = int(torch.multinomial(probs, 1))
            decision = 1 if idx == 0 else 0
            r = reward_for_decision(decision, ex)
            log_pi = torch.log_softmax(lps, dim=0)[idx]
            loss = -(r - baseline) * log_pi / len(batch)
            # backward per example: same gradient as a stacked mean, but each
            # graph is freed immediately — with real models, holding a full
            # batch of graphs would exhaust memory
            loss.backward()
            step_loss += loss.detach()
            rewards.append(r)
        opt.step()
        mean_reward = sum(rewards) / len(rewards)
        baseline = 0.9 * baseline + 0.1 * mean_reward
        if step % args.eval_every == 0 or step == args.steps:
            log_point(step, round(mean_reward, 4),
                      round(float(step_loss), 4))

    log_f.close()
    print(f"wrote {args.log_file} ({len(history)} points)")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        if hasattr(tokenizer, "hf_tokenizer"):
            tokenizer.hf_tokenizer.save_pretrained(args.save_dir)
        print(f"saved final checkpoint to {args.save_dir}/")
    return history


def make_checkpoint_backend(model_path: str, device: torch.device):
    """An eval_harness-compatible answer(prompt) backend for a local (or hub)
    causal-LM checkpoint: greedy decision by candidate scoring over the SAME
    natural-language prompts the API models see, replied as the one-word
    answer the harness parses. This puts a fine-tuned checkpoint in the same
    proof-of-life table as the model ladder."""
    model, tokenizer = build_model_and_tokenizer(model_path, seed=0)
    model.to(device)
    model.eval()

    def answer(prompt: str) -> str:
        with torch.no_grad():
            lps = candidate_logprobs(model, tokenizer, prompt, device)
        return "AUTHORIZED" if int(torch.argmax(lps)) == 0 else "UNAUTHORIZED"

    return answer


def eval_checkpoint(model_path: str, test_file: str, device: torch.device,
                    merge_results: str = None) -> dict:
    """Score a checkpoint on the natural-language benchmark via
    eval_harness.run_eval; optionally merge the entry into an existing
    proof-of-life results JSON under the name 'local:<path>'."""
    from eval_harness import print_summary, run_eval
    from trace_benchmark import load_traces

    traces = load_traces(test_file)
    out = run_eval(make_checkpoint_backend(model_path, device), traces)
    name = f"local:{model_path}"
    print_summary(name, out["metrics"])
    if merge_results:
        with open(merge_results) as f:
            results = json.load(f)
        results["backends"][name] = {"metrics": out["metrics"]}
        with open(merge_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"merged {name} into {merge_results}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a policy with the authorization verifier as reward.")
    ap.add_argument("--model", default="tiny",
                    help="'tiny' (offline smoke model) or a HF causal LM name")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-file", default="benchmark_train.jsonl")
    ap.add_argument("--test-file", default="benchmark_test.jsonl")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-max-actions", type=int, default=64)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps (auto picks cuda else cpu; "
                         "pass mps explicitly on Apple silicon)")
    ap.add_argument("--log-file", default="training_log.jsonl")
    ap.add_argument("--save-dir", default=None,
                    help="directory for the final model checkpoint "
                         "(HF save_pretrained format); omit to skip saving")
    ap.add_argument("--eval-checkpoint", default=None, metavar="PATH",
                    help="skip training; score this checkpoint on the "
                         "natural-language benchmark (eval_harness prompts "
                         "and parsing, same as the model ladder)")
    ap.add_argument("--merge-results", default=None, metavar="JSON",
                    help="with --eval-checkpoint: merge the entry into this "
                         "existing proof-of-life results file")
    args = ap.parse_args()

    if args.eval_checkpoint:
        eval_checkpoint(args.eval_checkpoint, args.test_file,
                        pick_device(args.device), args.merge_results)
        return

    train(args)

    if not torch.cuda.is_available():
        print(
            "\nNo CUDA GPU detected — this run served as the CPU smoke test "
            "and the training loop is verified end to end.\n"
            "READY FOR GPU: on a GPU machine, scale up with e.g.\n"
            "  PYTHONPATH=. python3 train_verifier_reward.py "
            "--model gpt2 --steps 500 --batch-size 16 --lr 1e-5 "
            "--eval-every 25")


if __name__ == "__main__":
    main()
