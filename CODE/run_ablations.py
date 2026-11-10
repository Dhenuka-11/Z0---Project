"""
Ablations and extra checks (Weeks 10-12). Every run uses the same 500 steps,
the same batches per seed and the same settings as the main experiment except
for the one thing named in its label.

    python run_ablations.py --list                 # show the matrix, run nothing
    python run_ablations.py --tier 1               # the must-have runs
    python run_ablations.py --tier 2               # the extra runs
    python run_ablations.py --only trace_static,trace_shuffled --seeds 0,1,2

Logs go to logs/abl/<name>_seed<N>.json and are skipped if already complete,
so the script can be stopped and restarted freely.
"""
import argparse
import json
import os
import time
from trace_scaled_zo import train

NUM_STEPS = 500
EVAL_STEPS = (0, NUM_STEPS)

# name: (tier, what it tests, train() keyword arguments)
CONFIGS = {
    # --- tier 1: the questions the report most needs answered ---------------
    "trace_static":   (1, "Refresh sigma only once (step 0). Is repeated refreshing hurting?",
                       dict(use_trace_scaling=True, trace_refresh_every=NUM_STEPS)),
    "trace_avg":      (1, "Running mean of each block's trace across refreshes (less noisy sigma).",
                       dict(use_trace_scaling=True, trace_average=True)),
    "trace_shuffled": (1, "Control: same sigma values, assigned to the WRONG blocks. Does curvature matter?",
                       dict(use_trace_scaling=True, sigma_mode="shuffled")),
    "trace_inv":      (1, "Alternate rule: sigma ~ 1/trace instead of 1/sqrt(trace).",
                       dict(use_trace_scaling=True, sigma_rule="inv")),
    "baseline_lr2e-6": (1, "Plain MeZO at a larger learning rate.",
                        dict(use_trace_scaling=False, lr=2e-6)),
    "trace_lr2e-6":   (1, "Trace-scaled at a larger learning rate.",
                       dict(use_trace_scaling=True, lr=2e-6)),
    "trace_lr5e-7":   (1, "Trace-scaled at a smaller learning rate.",
                       dict(use_trace_scaling=True, lr=5e-7)),
    # --- tier 2: sweeps and the Paper 3 check --------------------------------
    "trace_delta1":   (2, "Regularization constant delta = 1.",
                       dict(use_trace_scaling=True, delta=1.0)),
    "trace_delta100": (2, "Regularization constant delta = 100.",
                       dict(use_trace_scaling=True, delta=100.0)),
    "baseline_lr5e-7": (2, "Plain MeZO at a smaller learning rate.",
                        dict(use_trace_scaling=False, lr=5e-7)),
    "baseline_lr4e-6": (2, "Plain MeZO at a still larger learning rate.",
                        dict(use_trace_scaling=False, lr=4e-6)),
    "trace_lr4e-6":   (2, "Trace-scaled at a still larger learning rate.",
                       dict(use_trace_scaling=True, lr=4e-6)),
    "baseline_P4":    (2, "Paper 3 check: 4 perturbations per step, plain MeZO (4x forward passes).",
                       dict(use_trace_scaling=False, num_perturbations=4)),
    "trace_P4":       (2, "Paper 3 check: 4 perturbations per step, trace-scaled (4x forward passes).",
                       dict(use_trace_scaling=True, num_perturbations=4)),
}


def already_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return "eval" in json.load(f)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2], help="run one tier only")
    ap.add_argument("--only", help="comma-separated config names")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--out", default="logs/abl")
    args = ap.parse_args()

    names = list(CONFIGS)
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        unknown = [n for n in names if n not in CONFIGS]
        if unknown:
            raise SystemExit(f"unknown config(s): {unknown}")
    elif args.tier:
        names = [n for n in names if CONFIGS[n][0] == args.tier]
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.list:
        for n in names:
            tier, why, kw = CONFIGS[n]
            print(f"tier {tier}  {n:<16} {why}")
        print(f"\n{len(names)} configs x {len(seeds)} seeds = {len(names) * len(seeds)} runs")
        return

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for name in names:
        tier, why, kw = CONFIGS[name]
        for seed in seeds:
            path = f"{args.out}/{name}_seed{seed}.json"
            if already_done(path):
                print(f"skip {path} (already complete)")
                continue
            start = time.time()
            print(f"=== {name}, seed {seed} ===  {why}")
            train(num_steps=NUM_STEPS, seed=seed, log_path=path,
                  eval_steps=EVAL_STEPS,
                  log_block_details=kw.get("use_trace_scaling", False), **kw)
            print(f"finished in {(time.time() - start) / 60:.1f} min")
    print(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
