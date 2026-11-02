import os
import time
from trace_scaled_zo import train

SEEDS = [0, 1, 2, 3, 4]
NUM_STEPS = 500

os.makedirs("logs", exist_ok=True)

for seed in SEEDS:
    for use_trace, name in [(False, "baseline"), (True, "trace_scaled")]:
        path = f"logs/{name}_seed{seed}.json"
        if os.path.exists(path):
            print(f"skip {path} (already exists)")
            continue
        start = time.time()
        print(f"=== {name}, seed {seed} ===")
        train(use_trace_scaling=use_trace, num_steps=NUM_STEPS,
              seed=seed, log_path=path)
        print(f"finished in {(time.time() - start) / 60:.1f} min")