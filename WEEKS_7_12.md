# Weeks 7-12: what exists and what is still to run

Code written and unit-tested on a small fake model. "Run" items need a GPU and
the real OPT-125M; nothing in the Results column is filled until they finish.

| Week | Item (owner in the plan) | Code | Status |
|---|---|---|---|
| 7 | Logging of loss and per-block sigma (Dhenuka) | `train(log_block_details=True)` | code done |
| 7 | Held-out accuracy harness (Rishika) | `evaluate_accuracy`, `train(eval_steps=...)` | code done |
| 8 | Multi-seed baseline runs (Dhenuka) | `run_seeds.py` | **to run** (5 seeds) |
| 8 | Multi-seed trace-scaled runs (Rishika) | `run_seeds.py` | **to run** (5 seeds) |
| 9 | Steps-to-threshold and rolling-std metrics (Dhenuka) | `analyze_seeds.py` | code done |
| 9 | Paired significance test, comparison table (Rishika) | `analyze_seeds.py`, `summarize_runs.py` | code done; table after runs |
| 10 | Curvature diagnostic (Dhenuka) | `curvature_diagnostic.py` | code done; needs the new trace-scaled logs |
| 10 | Ablations: alternate rule, regularization sweep (Rishika) | `run_ablations.py` | **to run** (tier 1, then tier 2) |
| 11 | Findings note on diagnostic and ablations | -- | to write after the runs |
| 12 | Perturbation-count check from Paper 3 (Rishika) | `num_perturbations`, configs `baseline_P4`, `trace_P4` | **to run** (tier 2) |
| 12 | Second dataset (Dhenuka, optional) | -- | **not started** |
| 12 | Final experiment matrix locked | `python run_ablations.py --list` | after the runs |

## Notes
- Every new option in `train()` defaults to the old behaviour. On the fake
  model, default runs give losses identical to the previous version.
- The perturbation-count check is simplified: the same number of directions
  (4) for every block, not a separate count per block.
- The shuffled-sigma control (`trace_shuffled`) keeps the spread of sigma but
  gives each block another block's value. If it does as well as the real rule,
  curvature information is not what matters.
- The curvature diagnostic applies Paper 2's 2/Tr(H) bound to
  Tr(S H) = sum(sigma_i^2 * Tr_i). That step is our assumption, not Paper 2's.
