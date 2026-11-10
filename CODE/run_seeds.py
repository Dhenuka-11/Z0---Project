"""
Main experiment: plain MeZO vs trace-scaled MeZO, 5 seeds x 500 steps.

Logs loss every step, held-out SST-2 accuracy at steps 0, 250 and 500, and
(for trace-scaled runs) the per-block traces and sigmas at every refresh.

A log is skipped only if it already contains accuracy results, so logs from
the older version of this script (loss only) are re-run and overwritten.
Batches and perturbation seeds are unchanged, so the loss curves match.

    python run_seeds.py
"""
import json
import os
import time
from trace_scaled_zo import train

SEEDS = [0, 1, 2, 3, 4]
NUM_STEPS = 500
EVAL_STEPS = (0, 250, NUM_STEPS)

os.makedirs("logs", exist_ok=True)


def already_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return "eval" in json.load(f)
    except Exception:
        return False


if __name__ == "__main__":
    t0 = time.time()
    for seed in SEEDS:
        for use_trace, name in [(False, "baseline"), (True, "trace_scaled")]:
            path = f"logs/{name}_seed{seed}.json"
            if already_done(path):
                print(f"skip {path} (already complete)")
                continue
            start = time.time()
            print(f"=== {name}, seed {seed} ===")
            train(use_trace_scaling=use_trace, num_steps=NUM_STEPS, seed=seed,
                  log_path=path, eval_steps=EVAL_STEPS,
                  log_block_details=use_trace)
            print(f"finished in {(time.time() - start) / 60:.1f} min")
    print(f"all done in {(time.time() - t0) / 60:.1f} min")
