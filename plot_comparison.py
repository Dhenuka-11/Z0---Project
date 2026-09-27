import json
import matplotlib.pyplot as plt

with open("logs/baseline_run1.json") as f:
    baseline = json.load(f)
with open("logs/trace_scaled_run1.json") as f:
    trace_scaled = json.load(f)

plt.figure(figsize=(8, 5))
plt.plot(baseline["losses"], label="Plain MeZO (baseline)", color="tab:blue")
plt.plot(trace_scaled["losses"], label="Trace-scaled MeZO", color="tab:orange")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("MeZO baseline vs. trace-scaled: training loss")
plt.legend()
plt.tight_layout()
plt.savefig("logs/loss_comparison.png", dpi=150)
print("saved plot to logs/loss_comparison.png")
