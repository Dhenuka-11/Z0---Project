"""
Makes 4 graphs from the saved logs.

Run it from the folder that contains logs/   (e.g. Result_analyze/):
    python make_graphs.py
    python make_graphs.py --logs path/to/logs --out figs

Needs: numpy, matplotlib.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default="logs")
ap.add_argument("--out", default="figs")
ap.add_argument("--window", type=int, default=20, help="smoothing window (steps)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

PLAIN, TRACE = "tab:blue", "tab:red"


def load(pattern):
    runs = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def final_eval(run):
    return max(run["eval"], key=lambda e: e["step"])


def smooth(y, w):
    y = np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(w) / w, mode="valid")


base = load(f"{args.logs}/baseline_seed*.json")
trace = load(f"{args.logs}/trace_scaled_seed*.json")
print(f"plain MeZO runs: {len(base)}, trace-scaled runs: {len(trace)}")

# ---- Graph 1: training loss over the 500 steps -----------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for runs, color, label in [(base, PLAIN, "Plain MeZO"), (trace, TRACE, "Trace-scaled MeZO")]:
    curves = np.array([smooth(r["losses"], args.window) for r in runs])
    x = np.arange(curves.shape[1]) + args.window
    for c in curves:
        ax.plot(x, c, color=color, alpha=0.2, lw=1)
    ax.plot(x, curves.mean(axis=0), color=color, lw=2.2, label=f"{label} (mean of {len(runs)} seeds)")
ax.set_xlabel("training step")
ax.set_ylabel(f"training loss (moving average over {args.window} steps)")
ax.set_title("Training loss: plain MeZO vs trace-scaled MeZO")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{args.out}/graph1_training_loss.png", dpi=200)
plt.close(fig)

# ---- Graph 2: final accuracy per seed --------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
n = min(len(base), len(trace))
x = np.arange(n)
wd = 0.38
acc_b = [final_eval(r)["accuracy"] * 100 for r in base[:n]]
acc_t = [final_eval(r)["accuracy"] * 100 for r in trace[:n]]
ax.bar(x - wd / 2, acc_b, wd, color=PLAIN, label=f"Plain MeZO (mean {np.mean(acc_b):.1f}%)")
ax.bar(x + wd / 2, acc_t, wd, color=TRACE, label=f"Trace-scaled (mean {np.mean(acc_t):.1f}%)")
ax.axhline(50.92, color="gray", ls="--", lw=1)
ax.text(n - 0.5, 51.5, "untrained model: 50.9% (chance)", ha="right", color="gray", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {i}" for i in range(n)])
ax.set_ylim(40, 80)
ax.set_ylabel("validation accuracy (%)")
ax.set_title("Validation accuracy after 500 steps, per seed")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{args.out}/graph2_accuracy_per_seed.png", dpi=200)
plt.close(fig)

# ---- Graph 3: all configurations (main + ablations) ------------------------
groups = defaultdict(list)
for p in glob.glob(f"{args.logs}/*_seed*.json") + glob.glob(f"{args.logs}/abl/*_seed*.json"):
    name = re.match(r"(.+)_seed\d+\.json", os.path.basename(p)).group(1)
    with open(p) as f:
        r = json.load(f)
    if "eval" in r:
        groups[name].append(final_eval(r)["accuracy"] * 100)
rows = sorted(groups.items(), key=lambda kv: np.mean(kv[1]))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, (name, v) in enumerate(rows):
    color = PLAIN if name == "baseline" else TRACE if name == "trace_scaled" else "tab:gray"
    ax.barh(i, np.mean(v), color=color, alpha=0.8)
    ax.plot(v, [i] * len(v), "k.", ms=5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{n} (n={len(v)})" for n, v in rows], fontsize=8)
ax.axvline(50.92, color="gray", ls="--", lw=1)
ax.set_xlim(40, 80)
ax.set_xlabel("validation accuracy (%): bar = mean, dots = individual seeds")
ax.set_title("Main runs and ablations")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{args.out}/graph3_all_configs.png", dpi=200)
plt.close(fig)

# ---- Graph 4: where the sigma budget goes (trace-scaled runs) --------------
share_nan, share_neg, share_pos = [], [], []
par_nan, par_neg, par_pos = [], [], []
for r in trace:
    if "block_details" not in r:
        continue
    names = list(r["block_sizes"])
    n_i = np.array([r["block_sizes"][k] for k in names], dtype=float)
    N = n_i.sum()
    for rec in r["block_details"]:
        tr = np.array([np.nan if rec["traces"][k] is None else rec["traces"][k] for k in names])
        sg = np.array([rec["sigmas"][k] for k in names])
        isnan = np.isnan(tr)
        neg = ~isnan & (tr <= 0)
        pos = ~isnan & (tr > 0)
        for m, s_list, p_list in [(isnan, share_nan, par_nan), (neg, share_neg, par_neg), (pos, share_pos, par_pos)]:
            s_list.append((n_i[m] * sg[m] ** 2).sum() / N * 100)   # share of the sigma budget
            p_list.append(n_i[m].sum() / N * 100)                   # share of the parameters
fig, ax = plt.subplots(figsize=(7, 3.2))
labels = ["share of parameters", "share of σ budget (Σ n·σ²)"]
parts = [("positive trace", "tab:green", par_pos, share_pos),
         ("trace ≤ 0 (clamped)", "tab:red", par_neg, share_neg),
         ("trace NaN", "tab:gray", par_nan, share_nan)]
left = np.zeros(2)
for name, color, p, s in parts:
    vals = np.array([np.mean(p), np.mean(s)])
    ax.barh(labels, vals, left=left, color=color, label=name)
    for i, v in enumerate(vals):
        if v > 6:
            ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=8)
    left += vals
ax.set_xlim(0, 100)
ax.set_xlabel("%  (average over all trace refreshes of the trace-scaled runs)")
ax.set_title("Where the nudge budget goes")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(f"{args.out}/graph4_sigma_budget.png", dpi=200)
plt.close(fig)

print("saved 4 graphs in", args.out)
