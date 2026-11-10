"""
One table for every configuration: main runs and ablations together.

For each configuration (all seeds of one name) it reports the mean over seeds of
the last-20-step training loss and the final held-out accuracy, plus the paired
difference against the plain-MeZO baseline with the same seeds.

    python summarize_runs.py                 # reads logs/ and logs/abl/
    python summarize_runs.py --csv logs/summary.csv
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", default=["logs", "logs/abl"])
ap.add_argument("--csv", default="logs/summary.csv")
ap.add_argument("--window", type=int, default=20)
args = ap.parse_args()

groups = defaultdict(dict)  # name -> seed -> record
for d in args.dirs:
    for path in glob.glob(os.path.join(d, "*_seed*.json")):
        m = re.match(r"(.+)_seed(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        name = m.group(1)
        if name == "baseline_run1" or name.endswith("_run1"):
            continue
        with open(path) as f:
            groups[name][int(m.group(2))] = json.load(f)

if "baseline" not in groups:
    raise SystemExit("No baseline_seed*.json logs found.")


def last_loss(rec):
    return float(np.mean(rec["losses"][-args.window:]))


def final_acc(rec):
    ev = rec.get("eval")
    return max(ev, key=lambda e: e["step"])["accuracy"] if ev else None


base = groups["baseline"]
order = ["baseline", "trace_scaled"] + sorted(n for n in groups if n not in ("baseline", "trace_scaled"))
rows = []
print(f"{'config':<18}{'seeds':>6}{'last-%d loss' % args.window:>16}{'vs baseline':>14}{'final acc':>12}{'vs baseline':>14}")
for name in order:
    if name not in groups:
        continue
    g = groups[name]
    seeds = sorted(g)
    loss = np.array([last_loss(g[s]) for s in seeds])
    accs = [final_acc(g[s]) for s in seeds]
    acc = np.array(accs, float) if all(a is not None for a in accs) else None
    shared = [s for s in seeds if s in base]
    dl = (np.mean([last_loss(g[s]) - last_loss(base[s]) for s in shared])
          if shared and name != "baseline" else None)
    da = None
    if acc is not None and name != "baseline":
        pairs = [(final_acc(g[s]), final_acc(base[s])) for s in shared]
        if pairs and all(a is not None and b is not None for a, b in pairs):
            da = np.mean([a - b for a, b in pairs])
    rows.append((name, len(seeds), loss.mean(), dl, None if acc is None else acc.mean(), da))
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    print(f"{name:<18}{len(seeds):>6}{loss.mean():>16.3f}{f(dl, '+.3f'):>14}"
          f"{f(None if acc is None else acc.mean(), '.3f'):>12}{f(da, '+.3f'):>14}")

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
with open(args.csv, "w") as f:
    f.write("config,seeds,last_loss,loss_diff_vs_baseline,final_acc,acc_diff_vs_baseline\n")
    for r in rows:
        f.write(",".join("" if v is None else str(v) for v in r) + "\n")
print(f"\nsaved {args.csv}")
print("Differences are paired by seed (same batches). Training-batch loss is noisy; "
      "a few seeds cannot establish small differences.")
