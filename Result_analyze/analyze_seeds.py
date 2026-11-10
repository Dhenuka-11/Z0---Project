"""
Compare plain MeZO (baseline) vs trace-scaled MeZO from saved run logs.

Usage:
    python analyze_seeds.py                 # reads logs/*_seed*.json
    python analyze_seeds.py --threshold 3.0 # fix the steps-to-threshold level

Only files named baseline_seed<N>.json / trace_scaled_seed<N>.json are used,
so older single-run files such as baseline_run1.json are ignored.
Seeds are paired: a seed is included only if BOTH methods have a log for it.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default="logs")
parser.add_argument("--window", type=int, default=20,
                    help="window for smoothing / last-N average")
parser.add_argument("--threshold", type=float, default=None,
                    help="loss level for steps-to-threshold "
                         "(default: baseline mean smoothed loss at the midpoint)")
parser.add_argument("--out", default="logs/analysis_plot.png")
args = parser.parse_args()


def load(prefix):
    runs = {}
    for path in glob.glob(os.path.join(args.logs, f"{prefix}_seed*.json")):
        m = re.search(r"_seed(\d+)\.json$", path)
        if not m:
            continue
        with open(path) as f:
            runs[int(m.group(1))] = json.load(f)
    return runs


base_runs = load("baseline")
trace_runs = load("trace_scaled")
seeds = sorted(set(base_runs) & set(trace_runs))
if not seeds:
    raise SystemExit("No paired seeds found (need baseline_seedN.json and "
                     "trace_scaled_seedN.json for the same N).")

unpaired = sorted(set(base_runs) ^ set(trace_runs))
if unpaired:
    print(f"note: seeds without a partner run are ignored: {unpaired}")

n_steps = min(min(len(base_runs[s]["losses"]) for s in seeds),
              min(len(trace_runs[s]["losses"]) for s in seeds))
B = np.array([base_runs[s]["losses"][:n_steps] for s in seeds])
T = np.array([trace_runs[s]["losses"][:n_steps] for s in seeds])
W = args.window


def smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")  # length n - w + 1


# ---- metric 1: mean loss over the last W steps, per seed ------------------
last_B = B[:, -W:].mean(axis=1)
last_T = T[:, -W:].mean(axis=1)

# ---- metric 2: steps to threshold (on smoothed loss) ----------------------
SB = np.array([smooth(r, W) for r in B])
ST = np.array([smooth(r, W) for r in T])
if args.threshold is None:
    mid = SB.shape[1] // 2
    threshold = float(SB[:, mid].mean())
else:
    threshold = args.threshold


def steps_to(curve):
    hit = np.where(curve <= threshold)[0]
    # index i of the smoothed curve covers raw steps i..i+W-1
    return (hit[0] + W - 1) if len(hit) else None


stt_B = [steps_to(c) for c in SB]
stt_T = [steps_to(c) for c in ST]

# ---- metric 3: stability = mean rolling std of the raw loss ---------------
def rolling_std(x, w):
    return np.array([x[i:i + w].std() for i in range(len(x) - w + 1)]).mean()


std_B = np.array([rolling_std(r, W) for r in B])
std_T = np.array([rolling_std(r, W) for r in T])


def paired_report(name, a, b):
    diff = b - a  # trace_scaled minus baseline
    line = (f"{name:<34} baseline {a.mean():7.3f} ± {a.std(ddof=1) if len(a) > 1 else float('nan'):6.3f}"
            f" | trace {b.mean():7.3f} ± {b.std(ddof=1) if len(b) > 1 else float('nan'):6.3f}"
            f" | mean diff {diff.mean():+.3f}")
    if len(a) >= 2:
        t, p = stats.ttest_rel(b, a)
        line += f" | paired t-test p = {p:.3f}"
    print(line)


print(f"\nseeds used: {seeds}   steps per run: {n_steps}   window: {W}")
print(f"steps-to-threshold level: {threshold:.3f} "
      f"({'user-set' if args.threshold is not None else 'baseline mean at midpoint'})\n")
paired_report(f"mean loss, last {W} steps", last_B, last_T)
paired_report(f"mean rolling std (window {W})", std_B, std_T)

def final_accuracy(run):
    ev = run.get("eval")
    return max(ev, key=lambda e: e["step"])["accuracy"] if ev else None


acc_B = [final_accuracy(base_runs[s]) for s in seeds]
acc_T = [final_accuracy(trace_runs[s]) for s in seeds]
have_acc = all(a is not None for a in acc_B + acc_T)
if have_acc:
    paired_report("final held-out accuracy", np.array(acc_B), np.array(acc_T))
    start = base_runs[seeds[0]]["eval"][0]
    if start["step"] == 0:
        print(f"{'':<34} (before any training: accuracy {start['accuracy']:.3f}, "
              f"label-word loss {start['loss']:.3f})")
else:
    print("held-out accuracy: not in these logs (re-run with the updated run_seeds.py)")

reached_B = [s for s in stt_B if s is not None]
reached_T = [s for s in stt_T if s is not None]
print(f"\nsteps to threshold: baseline reached in {len(reached_B)}/{len(seeds)} runs "
      f"(median {np.median(reached_B) if reached_B else 'n/a'}), "
      f"trace-scaled reached in {len(reached_T)}/{len(seeds)} runs "
      f"(median {np.median(reached_T) if reached_T else 'n/a'})")
if len(reached_B) == len(seeds) and len(reached_T) == len(seeds):
    paired_report("steps to threshold", np.array(stt_B, float), np.array(stt_T, float))
else:
    print("  (paired test skipped: not every run reached the threshold)")

print("\nper-seed last-window loss:")
for s, a, b in zip(seeds, last_B, last_T):
    print(f"  seed {s}: baseline {a:.3f}   trace-scaled {b:.3f}   diff {b - a:+.3f}")

print("\nCaution: with only", len(seeds), "seeds a paired t-test has little power, "
      "and the rolling std includes the downward trend of the loss. "
      "Loss here is training-batch loss; held-out accuracy is reported above when the logs contain it.")

# ---- plot -----------------------------------------------------------------
x = np.arange(n_steps)
fig, ax = plt.subplots(figsize=(8, 5))
for data, label, color in [(B, "Plain MeZO", "tab:blue"),
                           (T, "Trace-scaled MeZO", "tab:orange")]:
    m, sd = data.mean(axis=0), data.std(axis=0)
    ax.plot(x, m, label=f"{label} (mean of {len(seeds)} seeds)", color=color)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.2)
ax.set_xlabel("Training step")
ax.set_ylabel("Training-batch loss")
ax.set_title("Mean loss across seeds (band = ±1 std)")
ax.legend()
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=150)
print(f"\nsaved plot to {args.out}")
