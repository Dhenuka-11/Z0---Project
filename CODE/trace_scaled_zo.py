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


def evaluate_accuracy(model, tokenizer, examples, max_examples=None):
    """
    Held-out evaluation (Week 7 harness). For each example, compare the
    model's next-token logits for the two label words and predict the larger
    one. Also returns the mean label-word loss on the same examples.
    `examples` can be a HuggingFace split or a list of dicts. Uses no
    randomness, so calling it never changes a training run.
    """
    label_token_ids = [tokenizer(w, add_special_tokens=False).input_ids[0]
                       for w in LABEL_WORDS]
    n = len(examples) if max_examples is None else min(max_examples, len(examples))
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i in range(n):
            ex = examples[i]
            input_ids = tokenizer(build_prompt(ex["sentence"]),
                                  return_tensors="pt").input_ids.to(DEVICE)
            logits = model(input_ids).logits[0, -1, :]
            pred = int(logits[label_token_ids].argmax().item())
            correct += int(pred == ex["label"])
            log_probs = F.log_softmax(logits, dim=-1)
            total_loss += -log_probs[label_token_ids[ex["label"]]].item()
    return {"accuracy": correct / n, "loss": total_loss / n, "n": n}


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


def mezo_step_multi(model, tokenizer, examples, blocks, sigmas, epsilon, lr,
                    num_perturbations):
    """
    Same as mezo_step, but averages P independent perturbation directions
    per step (the Week 12 perturbation-count check, motivated by Paper 3).
    Every direction is evaluated at the same starting weights, then all P
    updates are applied, each scaled by lr / P. Costs P times as many
    forward passes as mezo_step.
    """
    seeds = [random.randint(0, 2**31 - 1) for _ in range(num_perturbations)]

    def perturb(seed, scale):
        torch.manual_seed(seed)
        for name, params in blocks.items():
            sigma = sigmas[name]
            for _, p in params:
                z = torch.randn_like(p)
                p.add_(scale * sigma * z)

    loss_estimates = []
    grad_scalars = []
    with torch.no_grad():
        for s in seeds:
            perturb(s, +epsilon)
            loss_plus = compute_loss(model, tokenizer, examples)
            perturb(s, -epsilon)
            perturb(s, -epsilon)
            loss_minus = compute_loss(model, tokenizer, examples)
            perturb(s, +epsilon)  # back to the original weights
            grad_scalars.append((loss_plus - loss_minus) / (2 * epsilon))
            loss_estimates.append((loss_plus + loss_minus) / 2)

        for s, g in zip(seeds, grad_scalars):
            perturb(s, -(lr / num_perturbations) * g)

    return sum(loss_estimates) / num_perturbations


# ---------------------------------------------------------------------------
# Section 7: training loop wiring both variants
# ---------------------------------------------------------------------------

def _nan_to_none(x):
    return None if (x != x) else float(x)


def train(use_trace_scaling, num_steps=500, batch_size=8, lr=1e-6,
          epsilon=1e-3, trace_refresh_every=50, seed=0, log_path=None,
          sigma_rule="inv_sqrt", delta=1e-4, num_probes=15,
          sigma_mode="trace", trace_average=False, num_perturbations=1,
          eval_steps=(), eval_examples=None, log_block_details=False):
    """
    log_path: if given, results are saved as JSON to this path.

    Every option after log_path is for the Week 7-12 experiments, and each
    default reproduces the original behaviour exactly (same batches, same
    random numbers), so earlier logs stay comparable.

    sigma_rule:        "inv_sqrt" (1/sqrt(trace)) or "inv" (1/trace)
    delta:             constant added to the trace before inverting
    num_probes:        Hutchinson probes per block at each refresh
    sigma_mode:        "trace"    -> sigma set from the traces (the method)
                       "shuffled" -> same traces, but randomly reassigned to
                                     other blocks (control: keeps the spread
                                     of sigma, removes the link to curvature)
    trace_average:     use the running mean of each block's trace over all
                       refreshes so far, instead of only the latest estimate
    num_perturbations: directions averaged per step (1 = plain MeZO step)
    eval_steps:        steps at which to measure held-out accuracy; the value
                       num_steps means "after the last update"
    eval_examples:     max validation examples to score (None = all)
    log_block_details: save per-block traces and sigmas at every refresh
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = load_model_and_tokenizer()
    train_data, val_data = load_sst2()
    blocks = get_param_blocks(model)

    sigmas = {name: 1.0 for name in blocks}  # baseline default
    loss_log = []
    sigma_summary_log = []  # list of {step, min, mean, max}
    block_details = []
    eval_log = []
    trace_sum, trace_count = {}, {}

    batch_rng = random.Random(seed)
    probe_rng = random.Random(seed + 1000)
    control_rng = random.Random(seed + 2000)  # only used by sigma_mode="shuffled"
    eval_steps = set(eval_steps)

    def run_eval(step):
        result = evaluate_accuracy(model, tokenizer, val_data, eval_examples)
        result["step"] = step
        eval_log.append(result)
        print(f"step {step:4d}  val accuracy {result['accuracy']:.4f}  "
              f"val loss {result['loss']:.4f}")

    for step in range(num_steps):
        if step in eval_steps:
            run_eval(step)

        if use_trace_scaling and step % trace_refresh_every == 0:
            probe_examples = [train_data[i] for i in
                               probe_rng.sample(range(len(train_data)), 1)]
            traces = estimate_all_block_traces(model, tokenizer,
                                                probe_examples, blocks,
                                                num_probes=num_probes)
            traces_used = traces
            if trace_average:
                for n, t in traces.items():
                    if not math.isnan(t):
                        trace_sum[n] = trace_sum.get(n, 0.0) + t
                        trace_count[n] = trace_count.get(n, 0) + 1
                traces_used = {n: (trace_sum[n] / trace_count[n]
                                   if trace_count.get(n) else float("nan"))
                               for n in traces}
            if sigma_mode == "shuffled":
                names = list(traces_used.keys())
                values_shuffled = [traces_used[n] for n in names]
                control_rng.shuffle(values_shuffled)
                traces_used = dict(zip(names, values_shuffled))
            sigmas = compute_sigmas(traces_used, blocks, delta=delta,
                                    rule=sigma_rule)
            values = list(sigmas.values())
            sigma_summary_log.append({
                "step": step,
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            })
            if log_block_details:
                block_details.append({
                    "step": step,
                    "traces": {n: _nan_to_none(t) for n, t in traces.items()},
                    "traces_used": {n: _nan_to_none(t)
                                    for n, t in traces_used.items()},
                    "sigmas": dict(sigmas),
                    "nan_blocks": sum(1 for t in traces.values() if t != t),
                })

        batch_idx = batch_rng.sample(range(len(train_data)), batch_size)
        batch = [train_data[i] for i in batch_idx]
        random.seed(seed * 100000 + step)  # same perturbation seed in both methods

        if num_perturbations == 1:
            loss = mezo_step(model, tokenizer, batch, blocks, sigmas,
                              epsilon, lr)
        else:
            loss = mezo_step_multi(model, tokenizer, batch, blocks, sigmas,
                                    epsilon, lr, num_perturbations)
        loss_log.append(loss)

        if step % 20 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")

    if num_steps in eval_steps:
        run_eval(num_steps)

    if log_path is not None:
        import json
        record = {
            "use_trace_scaling": use_trace_scaling,
            "num_steps": num_steps,
            "lr": lr,
            "seed": seed,
            "losses": loss_log,
            "sigma_summary": sigma_summary_log,
            "config": {
                "batch_size": batch_size, "epsilon": epsilon,
                "trace_refresh_every": trace_refresh_every,
                "sigma_rule": sigma_rule, "delta": delta,
                "num_probes": num_probes, "sigma_mode": sigma_mode,
                "trace_average": trace_average,
                "num_perturbations": num_perturbations,
            },
        }
        if eval_log:
            record["eval"] = eval_log
        if log_block_details:
            record["block_details"] = block_details
            record["block_sizes"] = {n: sum(p.numel() for _, p in ps)
                                     for n, ps in blocks.items()}
        with open(log_path, "w") as f:
            json.dump(record, f, indent=2)
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
