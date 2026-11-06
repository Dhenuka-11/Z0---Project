"""
Curvature diagnostic (Week 10). Uses the per-block traces and sigmas that the
trace-scaled runs save at every refresh (log_block_details=True).

Two questions:

 1. Is the implementation doing what the rule says?
    With sigma_i ~ 1/sqrt(Tr_i + delta), the product sigma_i^2 * (Tr_i + delta)
    should be the same for every block. We report how close it is.

 2. How close is the learning rate to the stability limit?
    Paper 2 bounds the stable learning rate for zeroth-order methods by
    about 2 / Tr(H). With per-block scales the relevant quantity becomes
    Tr(S H) = sum_i sigma_i^2 * Tr_i, where S = diag(sigma^2). Applying the
    bound in that form to our method is OUR ASSUMPTION, not a result stated
    in Paper 2. We report  lr * Tr / 2  (1.0 would mean "at the bound"):
    once for plain MeZO (sigma = 1) and once for the trace-scaled sigmas.

Caveats printed with the results: traces come from a single training example
and a handful of probes per block, a few blocks have no usable estimate, and
the lower bound in Paper 2 needs the top eigenvalue, which we do not estimate.

    python curvature_diagnostic.py                     # logs/trace_scaled_seed*.json
    python curvature_diagnostic.py --glob "logs/abl/trace_inv_seed*.json"
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="logs/trace_scaled_seed*.json")
ap.add_argument("--out", default="logs/curvature_diagnostic.png")
ap.add_argument("--csv", default="logs/curvature_diagnostic.csv")
args = ap.parse_args()

files = sorted(glob.glob(args.glob))
runs = []
for path in files:
    with open(path) as f:
        rec = json.load(f)
    if rec.get("block_details"):
        runs.append((path, rec))
if not runs:
    raise SystemExit(f"No logs with block_details found for {args.glob!r}. "
                     "Re-run with the updated run_seeds.py.")

rows = []
for path, rec in runs:
    lr = rec["lr"]
    delta = rec.get("config", {}).get("delta", 1e-4)
    rule = rec.get("config", {}).get("sigma_rule", "inv_sqrt")
    shuffled = rec.get("config", {}).get("sigma_mode") == "shuffled"
    # In the shuffled control the sigmas were built from traces assigned to the
    # wrong blocks, so the true traces are the unshuffled "traces" field.
    trace_key = "traces" if shuffled else "traces_used"
    for d in rec["block_details"]:
        names = [n for n, t in d[trace_key].items() if t is not None]
        tr = np.array([d[trace_key][n] for n in names])
        sg = np.array([d["sigmas"][n] for n in names])
        tr_pos = np.clip(tr, 0.0, None)

        tr_plain = tr.sum()
        tr_scaled = (sg ** 2 * tr).sum()
        tr_plain_c = tr_pos.sum()
        tr_scaled_c = (sg ** 2 * tr_pos).sum()

        # invariant check on blocks with a positive trace
        pos = tr > 0
        if rule == "inv_sqrt":
            inv = sg[pos] ** 2 * (tr[pos] + delta)
        else:
            inv = sg[pos] * (tr[pos] + delta)
        spread = (float(inv.max() / inv.min())
                  if (pos.any() and not shuffled) else float("nan"))

        rows.append({
            "file": os.path.basename(path), "seed": rec["seed"], "step": d["step"],
            "usable_blocks": len(names), "nan_blocks": d["nan_blocks"],
            "negative_trace_blocks": int((tr < 0).sum()),
            "trace_plain": tr_plain, "trace_scaled": tr_scaled,
            "ratio_plain": lr * tr_plain / 2, "ratio_scaled": lr * tr_scaled / 2,
            "ratio_plain_clamped": lr * tr_plain_c / 2,
            "ratio_scaled_clamped": lr * tr_scaled_c / 2,
            "invariant_max_over_min": spread,
        })

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
with open(args.csv, "w") as f:
    cols = list(rows[0].keys())
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")


def avg(key):
    v = np.array([r[key] for r in rows], dtype=float)
    return np.nanmean(v), np.nanmin(v), np.nanmax(v)


lr = runs[0][1]["lr"]
print(f"\n{len(runs)} run(s), {len(rows)} refreshes, lr = {lr}")
print(f"usable blocks per refresh: mean {avg('usable_blocks')[0]:.1f}   "
      f"NaN blocks: mean {avg('nan_blocks')[0]:.1f}, max {avg('nan_blocks')[2]:.0f}   "
      f"blocks with negative trace: mean {avg('negative_trace_blocks')[0]:.1f}")
print(f"rule check: sigma^2*(trace+delta) max/min across blocks: "
      f"median {np.nanmedian([r['invariant_max_over_min'] for r in rows]):.4f} (1.0 = exact)")
print("\nlr * Tr / 2  (stability ratio; 1.0 = at Paper 2's upper bound)")
for label, key in [("plain MeZO, sigma = 1", "ratio_plain"),
                   ("trace-scaled sigmas", "ratio_scaled"),
                   ("plain MeZO, negative traces clamped to 0", "ratio_plain_clamped"),
                   ("trace-scaled, negative traces clamped to 0", "ratio_scaled_clamped")]:
    m, lo, hi = avg(key)
    print(f"  {label:<44} mean {m:9.4f}   range [{lo:.4f}, {hi:.4f}]")
print("\nCaveats: traces come from one training example per refresh; blocks "
      "without a usable estimate are left out of the sums; applying the "
      "2/Tr(H) bound to Tr(SH) is our assumption; the lower bound needs the "
      "top eigenvalue, which is not estimated here.")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

path0, rec0 = runs[0]
d0 = rec0["block_details"][0]
key0 = "traces" if rec0.get("config", {}).get("sigma_mode") == "shuffled" else "traces_used"
names = [n for n, t in d0[key0].items() if t is not None and t > 0]
t0 = np.array([d0[key0][n] for n in names])
s0 = np.array([d0["sigmas"][n] for n in names])
ax = axes[0]
ax.scatter(t0, s0, s=14)
xs = np.array([t0.min(), t0.max()])
ax.plot(xs, s0[np.argmax(t0)] * np.sqrt(t0.max() / xs), "k--", lw=1, label="slope -1/2")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("estimated block trace"); ax.set_ylabel("sigma")
ax.set_title(f"Sigma vs trace (seed {rec0['seed']}, step {d0['step']})")
ax.legend()

ax = axes[1]
for path, rec in runs:
    rs = [r for r in rows if r["file"] == os.path.basename(path)]
    ax.plot([r["step"] for r in rs], [r["ratio_plain"] for r in rs], "o-", color="tab:blue",
            alpha=0.6, label="plain (sigma=1)" if path == runs[0][0] else None)
    ax.plot([r["step"] for r in rs], [r["ratio_scaled"] for r in rs], "s-", color="tab:orange",
            alpha=0.6, label="trace-scaled" if path == runs[0][0] else None)
ax.axhline(1.0, color="k", ls="--", lw=1, label="Paper 2 upper bound")
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlabel("training step"); ax.set_ylabel("lr * Tr / 2")
ax.set_title("Distance from the stability bound")
ax.legend()

ax = axes[2]
by_step = {}
for r in rows:
    by_step.setdefault(r["step"], []).append(r["nan_blocks"])
ax.bar([str(k) for k in sorted(by_step)], [np.mean(by_step[k]) for k in sorted(by_step)])
ax.set_xlabel("refresh step"); ax.set_ylabel("blocks with no usable trace (mean)")
ax.set_title("Blocks falling back to sigma = 1")

fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"\nsaved {args.out} and {args.csv}")
