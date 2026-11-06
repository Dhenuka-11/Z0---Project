# Trace-Scaled Zeroth-Order Fine-tuning

CSCI 8000 project (Dhenuka Chelumalla, Rishika Reddy Egamamidi). We test whether
the per-block perturbation scale of MeZO can be set directly from an estimate of
each block's Hessian trace, instead of being learned, and compare it with plain
MeZO on SST-2 with OPT-125M.

## Setup
    pip install -r requirements.txt
A GPU is needed for the real runs (we used a Colab T4). The tests need no GPU.

## Files
| File | Purpose |
|---|---|
| `trace_scaled_zo.py` | model, data, block partition, Hutchinson trace estimator, sigma rule, MeZO step, training loop, held-out accuracy |
| `run_seeds.py` | main experiment: plain vs trace-scaled, 5 seeds x 500 steps |
| `run_ablations.py` | ablations and extra checks (`--list` shows the matrix) |
| `analyze_seeds.py` | plain vs trace-scaled: loss, accuracy, steps-to-threshold, stability, paired tests, plot |
| `summarize_runs.py` | one table over the main runs and all ablations |
| `curvature_diagnostic.py` | sigma vs trace, rule check, distance from the stability bound |
| `test_trace_estimator.py`, `test_week7_12.py` | unit tests |

## Reproduce
    python run_seeds.py                      # logs/*_seed*.json
    python run_ablations.py --tier 1         # logs/abl/*.json
    python analyze_seeds.py                  # logs/analysis_plot.png
    python summarize_runs.py                 # logs/summary.csv
    python curvature_diagnostic.py           # logs/curvature_diagnostic.png

## Tests
    python -m unittest test_trace_estimator -v    # downloads OPT-125M
    python -m unittest test_week7_12 -v           # tiny fake model, runs in seconds
