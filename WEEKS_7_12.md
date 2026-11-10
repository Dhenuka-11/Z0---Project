# Weeks 7-12: status

All code is unit-tested on a small fake model (`CODE/test_week7_12.py`).
Real runs were done on a free Colab T4 with OPT-125M on SST-2.

| Week | Item | Code | Status |
|---|---|---|---|
| 7 | Logging of loss and per-block sigma | `train(log_block_details=True)` | done, verified on the real model |
| 7 | Held-out accuracy harness | `evaluate_accuracy`, `train(eval_steps=...)` | done, verified on the real model |
| 8 | Multi-seed baseline runs | `CODE/run_seeds.py` | done (5 seeds) |
| 8 | Multi-seed trace-scaled runs | `CODE/run_seeds.py` | done (5 seeds) |
| 9 | Steps-to-threshold and rolling-std metrics | `Result_analyze/analyze_seeds.py` | done, run on the real logs |
| 9 | Paired significance test, comparison table | `analyze_seeds.py`, `summarize_runs.py` | done, run on the real logs |
| 10 | Curvature diagnostic | `Result_analyze/curvature_diagnostic.py` | done, run on the 5 trace-scaled logs |
| 10 | Ablation: alternate scaling rule | config `trace_inv` | done (3 seeds) |
| 10 | Ablation: sweep of the regularization constant delta | configs `trace_delta1`, `trace_delta100` | **not run** (tier 2) |
| 10 | Other tier-1 ablations (static, running mean, shuffled control, learning rates) | `CODE/run_ablations.py --tier 1` | done (7 configs x 3 seeds) |
| 11 | Findings note on diagnostic and ablations | -- | results are in `Result_analyze/logs/summary.csv` and the README; no separate note written |
| 12 | Perturbation-count check from Paper 3 | configs `baseline_P4`, `trace_P4` | **not run** (tier 2) |
| 12 | Second dataset (optional) | -- | **not started** |

## Notes
- Every new option in `train()` defaults to the old behaviour. On the fake
  model, default runs give losses identical to the previous version.
- The perturbation-count check is simplified: the same number of directions
  (4) for every block, not a separate count per block.
- The shuffled-sigma control (`trace_shuffled`) keeps the spread of sigma but
  gives each block another block's value. In our runs it did not do worse
  than the real rule in loss, so the curvature information does not appear to
  be what matters (3 seeds only).
- The curvature diagnostic applies Paper 2's 2/Tr(H) bound to
  Tr(S H) = sum(sigma_i^2 * Tr_i). That step is our assumption, not Paper 2's.
