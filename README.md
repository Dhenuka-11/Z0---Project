# Trace-Scaled Zeroth-Order Fine-tuning

CSCI 8000 project (Dhenuka Chelumalla, Rishika Reddy Egamamidi). We test whether
the per-block perturbation scale of MeZO can be set directly from an estimate of
each block's Hessian trace, instead of being learned, and compare it with plain
MeZO on SST-2 with OPT-125M.

This is our own implementation of a new idea built on MeZO (Malladi et al., 2023).
We do not reimplement or reproduce the three papers that motivated it
(ZO Fine-tuner; ZO Optimization at the Edge of Stability; the kernel view of ZO).

## Layout
| Folder / file | Contents |
|---|---|
| `CODE/` | everything needed to run the experiments |
| `CODE/trace_scaled_zo.py` | model, data, block partition, Hutchinson trace estimator, sigma rule, MeZO step, training loop, held-out accuracy |
| `CODE/run_seeds.py` | main experiment: plain vs trace-scaled, 5 seeds x 500 steps |
| `CODE/run_ablations.py` | ablations and extra checks (`--list` shows the full matrix) |
| `CODE/test_week7_12.py`, `CODE/test_trace_estimator.py` | unit tests |
| `CODE/requirements.txt` | Python packages |
| `Result_analyze/` | analysis scripts and all result files |
| `Result_analyze/analyze_seeds.py` | plain vs trace-scaled: loss, accuracy, steps-to-threshold, stability, paired tests, plot |
| `Result_analyze/summarize_runs.py` | one table over the main runs and all ablations |
| `Result_analyze/curvature_diagnostic.py` | sigma vs trace, rule check, distance from the stability bound |
| `Result_analyze/logs/` | run logs (`*_seed*.json`, `abl/`), `summary.csv`, plots |
| `WEEKS_7_12.md` | status of the Week 7-12 plan items |

`Result_analyze/logs/` also holds three files from an early first test run
(`baseline_run1.json`, `trace_scaled_run1.json`, `loss_comparison.png`). They come
from an older version of the code and are not part of the results below.

## Setup
    pip install -r CODE/requirements.txt
A GPU is needed for the real runs (we used a free Colab T4). The OPT-125M model
and the SST-2 data are downloaded automatically from Hugging Face. The unit
tests need no GPU.

## Reproduce
Run the experiments from `CODE/` (they write to `CODE/logs/`), then copy the
logs into `Result_analyze/logs/` and run the analysis from `Result_analyze/`.

    cd CODE
    python run_seeds.py                      # logs/*_seed*.json
    python run_ablations.py --tier 1         # logs/abl/*.json
    # copy CODE/logs/* into Result_analyze/logs/
    cd ../Result_analyze
    python analyze_seeds.py                  # logs/analysis_plot.png
    python summarize_runs.py                 # logs/summary.csv
    python curvature_diagnostic.py           # logs/curvature_diagnostic.png

## Tests
    cd CODE
    python -m unittest test_week7_12 -v           # tiny fake model, runs in seconds
    python -m unittest test_trace_estimator -v    # downloads OPT-125M

## Setup of the experiments
OPT-125M, SST-2 prompt "<sentence>\nSentiment:", prediction by comparing the
next-token scores of " great" and " terrible". 500 steps, batch size 8,
lr 1e-6, epsilon 1e-3, trace refreshed every 50 steps (15 Hutchinson probes,
one training example). Accuracy is measured on the 872 SST-2 validation
sentences. Both methods use the same batches for each seed.

## Results (see `Result_analyze/logs/summary.csv`)
Main comparison, 5 seeds:

| | Plain MeZO | Trace-scaled |
|---|---|---|
| Loss, last 20 steps | 0.968 | 3.184 (paired t-test p = 0.007) |
| Final accuracy | 0.711 | 0.578 (p = 0.051) |
| Reached loss target 1.767 | 5 of 5 runs (median step 253) | 0 of 5 |

Plain MeZO was better on average. Trace-scaled was better in one seed (seed 3:
0.742 vs 0.687 accuracy) and stayed at about chance level (0.509-0.510) in
three seeds (0, 1 and 2).

Tier-1 ablations of trace-scaled MeZO, 3 seeds each (seeds 0-2):

| Variant | Loss | Accuracy |
|---|---|---|
| plain MeZO (same seeds) | 0.968 | 0.698 |
| trace-scaled (same seeds) | 3.047 | 0.510 |
| static sigma (computed once) | 1.119 | 0.671 |
| running-mean traces | 2.883 | 0.617 |
| shuffled traces (control) | 1.313 | 0.521 |
| sigma proportional to 1/trace | 2.472 | 0.559 |
| trace-scaled, lr 2e-6 | 2.418 | 0.509 |
| trace-scaled, lr 5e-7 | 3.903 | 0.554 |
| plain MeZO, lr 2e-6 | 0.905 | 0.614 |

## Limits
One small model, one dataset, 500 steps. The ablations use 3 seeds and 872
validation sentences, so small differences between variants are not reliable.
The trace is estimated from a single training example per refresh, and about 5
of 99 blocks give NaN and about 11 give negative values at each refresh. We
believe the noisy estimate is a likely cause of the poor result, but we have
not tested that directly. The stability check applies Paper 2's bound to the
scaled sum, which is our assumption, not Paper 2's. Not run: the tier-2
ablations (delta sweep, extra learning rates, 4 perturbations per step) and a
second dataset.
