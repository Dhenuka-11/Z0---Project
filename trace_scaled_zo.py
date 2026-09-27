"""
Trace-Scaled Zeroth-Order Fine-tuning — starting skeleton.

This is a STARTING POINT, not finished, tested code. You will need to debug
it in your own environment (it needs internet access to download the model
and dataset from HuggingFace, and a GPU is strongly recommended). Read
PROJECT_GUIDE.md alongside this file — each numbered section below matches
a step in that guide.

Install first:
    pip install torch transformers datasets accelerate scipy
"""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Section 1: model, tokenizer, and data
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/opt-125m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SST-2 prompt template, in the style used by the MeZO paper: turn each
# example into a short prompt and compare the model's probability for the
# two label words.
LABEL_WORDS = [" terrible", " great"]  # index 0 = negative, 1 = positive


def build_prompt(sentence):
    return f"{sentence}\nSentiment:"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(DEVICE)
    model.eval()  # we never call .train() — no dropout etc. during ZO steps
    return model, tokenizer


def load_sst2():
    ds = load_dataset("nyu-mll/glue", "sst2")
    return ds["train"], ds["validation"]


def compute_loss(model, tokenizer, examples):
    """
    Compute the average classification loss over a small batch of SST-2
    examples, using the label-word-probability trick. `examples` is a list
    of dicts with 'sentence' and 'label' (0 or 1) keys.
    """
    total_loss = 0.0
    for ex in examples:
        prompt = build_prompt(ex["sentence"])
        label_word = LABEL_WORDS[ex["label"]]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        label_ids = tokenizer(label_word, add_special_tokens=False,
                               return_tensors="pt").input_ids.to(DEVICE)
        # Score the label word's first token as the next-token prediction.
        with torch.no_grad():
            out = model(input_ids)
            next_token_logits = out.logits[0, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            loss = -log_probs[label_ids[0, 0]]
        total_loss += loss.item()
    return total_loss / len(examples)


# ---------------------------------------------------------------------------
# Section 2 & 3: block partition
# ---------------------------------------------------------------------------

def get_param_blocks(model):
    """
    Group the model's parameters into named blocks. This simple version
    groups by top-level module name (e.g. each transformer layer becomes
    one block). Adjust the grouping logic to whatever granularity you want.
    Returns: dict {block_name: [ (param_name, param_tensor), ... ]}
    """
    blocks = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # crude grouping: use everything up to the 3rd dot as the block key
        # e.g. "model.decoder.layers.4.self_attn.q_proj.weight"
        #      -> "model.decoder.layers.4"
        block_key = name.rsplit(".", 1)[0] if "." in name else name
        blocks.setdefault(block_key, []).append((name, param))
    return blocks


# ---------------------------------------------------------------------------
# Section 4: Hutchinson trace estimator (periodic, uses real backward passes)
# ---------------------------------------------------------------------------

def estimate_block_trace(model, tokenizer, examples, block_params,
                          num_probes=15):
    """
    Estimate the Hessian trace for ONE block using Hutchinson's estimator.
    block_params: list of (name, tensor) for this block only.
    This function DOES use backward() — only call it every K steps, never
    inside the main MeZO update loop.
    """
    params = [p for _, p in block_params]
    for p in params:
        p.requires_grad_(True)

    model.zero_grad()
    # Need actual forward pass with grad tracking here (unlike compute_loss,
    # which runs under torch.no_grad() for the MeZO steps).
    ex = examples[0]  # single example for speed; average over more if noisy
    prompt = build_prompt(ex["sentence"])
    label_word = LABEL_WORDS[ex["label"]]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    label_ids = tokenizer(label_word, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(DEVICE)
    out = model(input_ids)
    next_token_logits = out.logits[0, -1, :]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    loss = -log_probs[label_ids[0, 0]]

    grads = torch.autograd.grad(loss, params, create_graph=True)

    trace_sum = 0.0
    valid_probes = 0
    for _ in range(num_probes):
        zs = [torch.randint(0, 2, p.shape, device=DEVICE).float() * 2 - 1
              for p in params]  # Rademacher: entries are +1 or -1
        gz = sum((g * z).sum() for g, z in zip(grads, zs))
        Hz = torch.autograd.grad(gz, params, retain_graph=True)
        probe_value = sum((z * hz).sum().item() for z, hz in zip(zs, Hz))
        if not (probe_value != probe_value):  # skip NaN probes (NaN != NaN is True)
            trace_sum += probe_value
            valid_probes += 1

    for p in params:
        p.requires_grad_(False)

    if valid_probes == 0:
        return float("nan")
    return trace_sum / valid_probes


def estimate_all_block_traces(model, tokenizer, examples, blocks,
                               num_probes=15):
    return {name: estimate_block_trace(model, tokenizer, examples, params,
                                        num_probes=num_probes)
            for name, params in blocks.items()}


# ---------------------------------------------------------------------------
# Section 5: perturbation scaling rule
# ---------------------------------------------------------------------------

def compute_sigmas(trace_estimates, blocks, delta=1e-4, rule="inv_sqrt"):
    """
    Compute a normalized sigma_i per block from trace estimates.
    rule: "inv_sqrt" -> sigma_i ~ 1/sqrt(Tr(H_i) + delta)
          "inv"      -> sigma_i ~ 1/(Tr(H_i) + delta)
    Normalization follows Paper 1's fixed-budget rule:
        sum_i (param_count_i * sigma_i^2) == total_param_count
    """
    raw = {}
    for name, trace in trace_estimates.items():
        if math.isnan(trace):
            raw[name] = 1.0  # fallback for blocks where the trace estimate is unstable (e.g. embeddings)
        else:
            t = max(trace, 0.0) + delta
            raw[name] = (1.0 / math.sqrt(t)) if rule == "inv_sqrt" else (1.0 / t)

    param_counts = {name: sum(p.numel() for _, p in params)
                     for name, params in blocks.items()}
    total_params = sum(param_counts.values())

    weighted_sq_sum = sum(param_counts[name] * (raw[name] ** 2)
                            for name in raw)
    scale = math.sqrt(total_params / weighted_sq_sum)

    return {name: raw[name] * scale for name in raw}


# ---------------------------------------------------------------------------
# Section 6: MeZO training step (shared by both variants)
# ---------------------------------------------------------------------------

def mezo_step(model, tokenizer, examples, blocks, sigmas, epsilon, lr):
    """
    One MeZO update step, with a per-block sigma (pass a dict of all-ones
    for the plain baseline). Uses the seed-reuse trick so the noise vector
    is never fully materialized twice.
    """
    seed = random.randint(0, 2**31 - 1)

    def apply_perturbation(sign):
        torch.manual_seed(seed)
        for name, params in blocks.items():
            sigma = sigmas[name]
            for _, p in params:
                z = torch.randn_like(p)
                p.add_(sign * epsilon * sigma * z)

    with torch.no_grad():
        apply_perturbation(+1)
        loss_plus = compute_loss(model, tokenizer, examples)
        apply_perturbation(-1)  # undo +eps, so this brings us to -eps
        apply_perturbation(-1)
        loss_minus = compute_loss(model, tokenizer, examples)
        apply_perturbation(+1)  # undo -eps, back to original theta

        grad_scalar = (loss_plus - loss_minus) / (2 * epsilon)

        torch.manual_seed(seed)
        for name, params in blocks.items():
            sigma = sigmas[name]
            for _, p in params:
                z = torch.randn_like(p)
                p.add_(-lr * grad_scalar * sigma * z)

    return (loss_plus + loss_minus) / 2  # rough loss estimate for logging


# ---------------------------------------------------------------------------
# Section 7: training loop wiring both variants
# ---------------------------------------------------------------------------

def train(use_trace_scaling, num_steps=500, batch_size=8, lr=1e-6,
          epsilon=1e-3, trace_refresh_every=50, seed=0, log_path=None):
    """
    log_path: if given, results are saved as JSON to this path, containing
    per-step loss and, at each trace-refresh point, a summary of the sigma
    values used (min/mean/max across blocks) -- not the full per-block
    detail (that is added separately for the Week 9 diagnostic).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = load_model_and_tokenizer()
    train_data, _ = load_sst2()
    blocks = get_param_blocks(model)

    sigmas = {name: 1.0 for name in blocks}  # baseline default
    loss_log = []
    sigma_summary_log = []  # list of {step, min, mean, max}

    for step in range(num_steps):
        if use_trace_scaling and step % trace_refresh_every == 0:
            probe_examples = [train_data[i] for i in
                               random.sample(range(len(train_data)), 1)]
            traces = estimate_all_block_traces(model, tokenizer,
                                                probe_examples, blocks)
            sigmas = compute_sigmas(traces, blocks)
            values = list(sigmas.values())
            sigma_summary_log.append({
                "step": step,
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            })

        batch_idx = random.sample(range(len(train_data)), batch_size)
        batch = [train_data[i] for i in batch_idx]

        loss = mezo_step(model, tokenizer, batch, blocks, sigmas,
                          epsilon, lr)
        loss_log.append(loss)

        if step % 20 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")

    if log_path is not None:
        import json
        with open(log_path, "w") as f:
            json.dump({
                "use_trace_scaling": use_trace_scaling,
                "num_steps": num_steps,
                "lr": lr,
                "seed": seed,
                "losses": loss_log,
                "sigma_summary": sigma_summary_log,
            }, f, indent=2)
        print(f"saved log to {log_path}")

    return loss_log



def test_trace_estimator_toy():
    """
    Standalone sanity check for the Hutchinson trace estimator, using a known
    2x2 quadratic loss (loss = 0.5 * x^T H x) instead of the real model.
    Exact trace of H = [[4,1],[1,2]] is 6.
    """
    H = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    x = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

    def toy_loss():
        return 0.5 * x @ H @ x

    loss = toy_loss()
    grads = torch.autograd.grad(loss, [x], create_graph=True)

    num_probes = 2000
    trace_sum = 0.0
    for _ in range(num_probes):
        z = torch.randint(0, 2, x.shape).float() * 2 - 1
        gz = sum((g * zz).sum() for g, zz in zip(grads, [z]))
        Hz = torch.autograd.grad(gz, [x], retain_graph=True)
        trace_sum += sum((zz * hz).sum().item() for zz, hz in zip([z], Hz))

    estimate = trace_sum / num_probes
    print(f"estimated trace: {estimate:.4f}  (exact trace: 6.0)")

if __name__ == "__main__":
    test_trace_estimator_toy()
    print("Running plain MeZO baseline (sigma_i = 1 everywhere)...")
    baseline_losses = train(use_trace_scaling=False, num_steps=200)

    print("\nRunning trace-scaled MeZO...")
    trace_scaled_losses = train(use_trace_scaling=True, num_steps=200)
