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
biasing the initial policy to one decision. Two objectives:

  - default: sampled REINFORCE with a running-mean baseline (a decision is
    sampled from the two-way softmax; high estimator variance — documented
    to collapse into blanket policies at small scale);
  - --exact-pg: the closed-form expected reward E[r] = pi(A)r(A)+pi(R)r(R),
    computable because the verifier prices BOTH decisions; zero estimator
    variance, deterministic per seed, corners unreachable by noise.

Note a deliberate prompt-format split: training and validation curves use
the compact prompt above, while --eval-checkpoint scores the natural-
language eval_harness prompts (identical to the API-model ladder) — the
two scales are not directly comparable and both should be reported.
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


def build_model_and_tokenizer(name: str, seed: int,
                              attn_implementation: str = None):
    """`tiny` builds a local random-weight model (offline). Anything else is
    resolved through Hugging Face (may download on first use).

    attn_implementation="eager" is REQUIRED on MPS: the fused SDPA backward
    kernel produces NaN gradients for this model family on Apple silicon
    (empirically: all-NaN grads on the very first backward)."""
    torch.manual_seed(seed)
    kw = {"attn_implementation": attn_implementation} \
        if attn_implementation else {}
    if name == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=256, n_positions=1024,
                            n_embd=32, n_layer=2, n_head=2)
        return GPT2LMHeadModel(config), ByteTokenizer()
    if os.path.isdir(name) and not os.path.exists(
            os.path.join(name, "tokenizer_config.json")):
        # a saved tiny-model checkpoint: weights only, byte tokenizer
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(name, **kw), ByteTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kw)

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


def load_examples(path: str, prompt_style: str = "compact") -> list:
    """One example per action: the prompt plus the verifier objects needed to
    compute the reward at decision time.

    prompt_style="nl" renders the natural-language eval_harness prompt (the
    exact one the API-model ladder is scored on) instead of the terse
    compact prompt, so training and the final ladder evaluation share a
    format and the comparison is apples-to-apples."""
    build = None
    if prompt_style == "nl":
        from eval_harness import build_prompt as build
    elif prompt_style != "compact":
        raise ValueError(f"unknown prompt_style {prompt_style!r}")
    examples = []
    for tr in load_traces(path):
        root, chain, actions = trace_to_objects(tr)
        for act, aj in zip(actions, tr["actions"]):
            examples.append({
                "prompt": build(tr, aj) if build else compact_prompt(tr, aj),
                "action": act,
                "chain": chain,
                "root": root,
                "scenario_class": tr["scenario_class"],
            })
    return examples


def reward_for_decision(decision: int, example: dict,
                        class_weights: dict = None) -> float:
    """+1 iff the decision matches the verifier's verdict, else -1. The
    verifier is the sole reward source — labels stored in the corpus are
    never read here. Optional class_weights ({verdict: weight}) scale the
    reward by the action's true class, removing the majority-class
    attractor that drives always-refuse collapse under label imbalance."""
    verdict = label_action(example["action"], example["chain"], example["root"])
    r = 1.0 if decision == verdict else -1.0
    if class_weights:
        r *= class_weights[verdict]
    return r


def compute_class_weights(examples: list) -> dict:
    """Inverse-frequency weights over the verifier's live verdicts (never
    the stored labels), normalized so a balanced corpus gives weight 1."""
    labels = [label_action(ex["action"], ex["chain"], ex["root"])
              for ex in examples]
    n1 = sum(labels)
    n0 = len(labels) - n1
    return {0: len(labels) / (2 * n0), 1: len(labels) / (2 * n1)}


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
    if device.type == "mps":
        # an 80-forward eval sweep leaves the MPS allocator far beyond its
        # recommended working set; under that pressure the Metal kernels
        # can silently produce NaN instead of erroring. Release the cache.
        torch.mps.empty_cache()
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
    attn = "eager" if device.type == "mps" else None
    # Warm start: initialize the policy from a CE checkpoint instead of the
    # base model (the established supervised-warm-start-then-RL recipe).
    model_src = args.warm_start_from or args.model
    model, tokenizer = build_model_and_tokenizer(model_src, args.seed, attn)
    model.to(device)
    if args.warm_start_from:
        # Released CE checkpoints are saved fp16; fp16 is stable for
        # forward-only eval but its narrow range overflows under the training
        # backward/optimizer, producing non-finite scores. Train in fp32.
        model = model.float()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # KL anchor: a FROZEN reference policy (a second copy of the warm-start
    # checkpoint) that the RL update is penalized for drifting away from, so
    # refinement stays near the good warm policy instead of wandering into a
    # corner. Only built when --kl-coef > 0 (keeps the no-KL path unchanged).
    ref_model = None
    if args.kl_coef > 0:
        ref_src = args.warm_start_from or args.model
        ref_model, _ = build_model_and_tokenizer(ref_src, args.seed, attn)
        ref_model.to(device)
        if args.warm_start_from:
            ref_model = ref_model.float()  # match the fp32 policy for a valid KL
        ref_model.eval()
        ref_model.requires_grad_(False)

    train_examples = load_examples(args.train_file, args.prompt_style)
    class_weights = (compute_class_weights(train_examples)
                     if args.balance_reward else None)
    if class_weights:
        print(f"class-balanced rewards: {class_weights}")
    # Fixed-seed shuffle before truncating: a plain [:n] slice of the
    # trace-id-sorted list would silently drop whole scenario classes, and
    # shuffling by args.seed would monitor each seed on a different subset —
    # a fixed eval seed keeps the curves comparable across training seeds.
    eval_examples = load_examples(args.test_file, args.prompt_style)
    random.Random(0).shuffle(eval_examples)
    eval_examples = eval_examples[: args.eval_max_actions]
    print(f"device={device.type} model={args.model} "
          f"train_actions={len(train_examples)} eval_actions={len(eval_examples)}")

    history = []
    baseline = 0.0
    nan_steps_skipped = 0
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
            "nan_steps_skipped": nan_steps_skipped,
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
            if not torch.isfinite(lps).all():
                if any(not torch.isfinite(p).all()
                       for p in model.parameters()):
                    raise RuntimeError(
                        "model parameters are non-finite — a NaN update got "
                        "through; rerun (on MPS, keep the cache-release and "
                        "skip guards enabled)")
                raise RuntimeError(f"non-finite candidate scores at step "
                                   f"{step}: {lps.tolist()}")
            # KL penalty toward the frozen reference policy, over the two-way
            # decision softmax: KL(pi || pi_ref) = sum_d pi(d) (logpi - logref).
            # Added to whichever objective loss is used below. 0 when disabled.
            kl_term = 0.0
            if ref_model is not None:
                with torch.no_grad():
                    ref_lps = candidate_logprobs(
                        ref_model, tokenizer, ex["prompt"], device)
                ref_logp = torch.log_softmax(ref_lps, dim=0)
                cur_logp = torch.log_softmax(lps, dim=0)
                kl = (cur_logp.exp() * (cur_logp - ref_logp)).sum()
                kl_term = args.kl_coef * kl / len(batch)
            if args.ce_loss:
                # Verifier cross-entropy: the verdict is the target decision
                # and the loss is -log pi(target). Its gradient is
                # (pi(target) - 1) — which stays STRONG when the model is
                # confidently wrong, unlike the expected-reward objective
                # whose gradient ~ pi(A)pi(R) vanishes at the corners and
                # merely oscillates. Still verifier-only: label_action is
                # the sole supervision, never the stored corpus label. This
                # is the verifier signal used as a target rather than a
                # scalar reward (reward-argmax imitation).
                verdict = label_action(ex["action"], ex["chain"], ex["root"])
                target_idx = 0 if verdict == 1 else 1  # idx 0 = AUTHORIZE
                logp = torch.log_softmax(lps, dim=0)
                w = class_weights[verdict] if class_weights else 1.0
                loss = -w * logp[target_idx] / len(batch) + kl_term
                loss.backward()
                step_loss += loss.detach()
                with torch.no_grad():
                    pr = torch.softmax(lps, dim=0)
                    exp_r = (pr[0] * reward_for_decision(1, ex, class_weights)
                             + pr[1] * reward_for_decision(0, ex, class_weights))
                rewards.append(float(exp_r))
                continue
            if args.exact_pg:
                # Exact two-action policy gradient: with only two decisions
                # and the verifier able to price BOTH (label_action depends
                # only on the verdict), the expected reward E[r] =
                # pi(A)*r(A) + pi(R)*r(R) is computable in closed form.
                # Maximizing it directly is the zero-variance version of
                # REINFORCE. NOTE: its gradient ~ pi(A)pi(R) saturates at the
                # corners, so it oscillates rather than converging — use
                # --ce-loss for reliable convergence; this mode is kept as
                # the documented ablation.
                probs_g = torch.softmax(lps, dim=0)
                r_auth = reward_for_decision(1, ex, class_weights)
                r_ref = reward_for_decision(0, ex, class_weights)
                exp_r = probs_g[0] * r_auth + probs_g[1] * r_ref
                loss = -exp_r / len(batch) + kl_term
                if args.entropy_beta > 0:
                    logp = torch.log_softmax(lps, dim=0)
                    loss = loss - args.entropy_beta * \
                        (-(logp.exp() * logp).sum()) / len(batch)
                loss.backward()
                step_loss += loss.detach()
                rewards.append(float(exp_r.detach()))
                continue
            probs = torch.softmax(lps.detach(), dim=0)
            idx = int(torch.multinomial(probs, 1))
            decision = 1 if idx == 0 else 0
            r = reward_for_decision(decision, ex, class_weights)
            log_pi = torch.log_softmax(lps, dim=0)[idx]
            loss = -(r - baseline) * log_pi / len(batch) + kl_term
            if args.entropy_beta > 0:
                # entropy bonus keeps P(minority decision) alive so the
                # policy keeps exploring instead of locking into refusal
                logp = torch.log_softmax(lps, dim=0)
                entropy = -(logp.exp() * logp).sum()
                loss = loss - args.entropy_beta * entropy / len(batch)
            # backward per example: same gradient as a stacked mean, but each
            # graph is freed immediately — with real models, holding a full
            # batch of graphs would exhaust memory
            loss.backward()
            step_loss += loss.detach()
            rewards.append(r)
        if not all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in model.parameters()):
            # flaky-kernel guard (observed on MPS under memory pressure):
            # drop this step's update rather than poisoning the weights
            nan_steps_skipped += 1
            print(f"step {step}: non-finite gradients — update skipped "
                  f"({nan_steps_skipped} total)")
            opt.zero_grad()
            continue
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_norm)
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
    attn = "eager" if device.type == "mps" else None
    model, tokenizer = build_model_and_tokenizer(model_path, seed=0,
                                                 attn_implementation=attn)
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
    # key by model AND test file so evaluating one checkpoint on several
    # test sets (committed/fresh/per-domain) doesn't overwrite prior entries
    name = f"local:{model_path}::{os.path.basename(test_file)}"
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
    ap.add_argument("--clip-grad-norm", type=float, default=1.0,
                    help="max gradient norm per step (0 disables clipping)")
    ap.add_argument("--entropy-beta", type=float, default=0.0,
                    help="entropy-bonus coefficient (e.g. 0.01) to prevent "
                         "collapse into a single decision")
    ap.add_argument("--balance-reward", action="store_true",
                    help="scale rewards by inverse class frequency of the "
                         "verifier's verdicts (removes the majority-class "
                         "always-refuse attractor)")
    ap.add_argument("--exact-pg", action="store_true",
                    help="closed-form expected-reward objective (ablation): "
                         "zero sampling variance but a corner-saturating "
                         "gradient, so it oscillates — prefer --ce-loss")
    ap.add_argument("--ce-loss", action="store_true",
                    help="verifier cross-entropy: train -log pi(verdict). "
                         "Non-saturating gradient, converges cleanly. The "
                         "recommended objective for real accuracy.")
    ap.add_argument("--prompt-style", default="compact",
                    choices=("compact", "nl"),
                    help="'nl' trains on the natural-language ladder prompts "
                         "so training and final ladder eval share a format")
    ap.add_argument("--warm-start-from", default=None, metavar="PATH",
                    help="initialize the policy from a CE checkpoint "
                         "(dir or HF id) instead of the base model — the "
                         "supervised-warm-start-then-RL recipe")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL penalty coefficient toward a frozen reference "
                         "policy (the warm-start checkpoint), keeping RL "
                         "refinement near the warm policy (0 disables)")
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

    if args.ce_loss and args.exact_pg:
        ap.error("--ce-loss and --exact-pg are mutually exclusive objectives")
    if args.kl_coef < 0:
        ap.error("--kl-coef must be >= 0")

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
