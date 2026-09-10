# FWC color-direction pilot

## Experiment goal

Verify the already specified rule that higher feature contribution maps toward
red and lower contribution maps toward gray, then run one fixed PU T2 seed to
measure the consequence without changing the model, data, loss, or optimizer.

## Code version

- repo: `pangyu5451/AEI`
- local path: `D:\pycharm\AEI\DG-for-RMFD-main\AGG_FWC`
- commit: `9f5bed1` for the color-direction correction
- branch: `master`

## Fixed protocol

- dataset: PU at `D:\datasets\PU`
- requested transfer task: `[[0, 1, 2], 3]`
- actual source conditions in the legacy manifest: `N15_M01_F10`, `N15_M07_F04`, `N15_M07_F10`
- actual target condition in the legacy manifest: `N09_M07_F10`
- seed: `2026`
- epoch: `60`
- CUDA: required; RTX 3080 Ti, PyTorch `2.11.0+cu128`
- protocol label: `paper_condition`, with `bearing_disjoint=false`
- task semantics: current manifest uses `bearing_class:<bearing_id>`, not physical fault types

## Single change

The ranking used by `assign_items` changed from descending contribution to
ascending contribution before applying the fixed quotas. Thus the lowest
contribution features enter the gray quota and the highest contribution
features enter the red quota. Legal sizes, price/utility calculation, red
eligibility, collectibles, and packing were not changed.

## Results

| Metric | Previous FWC run | Color-direction pilot |
|---|---:|---:|
| Accuracy | 0.2857 | 0.2071 |
| Macro-F1 | 0.1613 | 0.1451 |
| Target evaluation count | 1 | 1 |
| CUDA fallback | false | false |

## Interpretation

The semantic rule correction passed its unit regression test but did not improve
this single target-condition result. This is not evidence that the rule is
invalid: the run is a `paper_condition` bearing-identity task rather than a
strict unseen-bearing physical-fault task, and the requested `transfer_task`
was not propagated by the legacy hardcoded manifest. The changed packing gate
can also alter the frozen classifier's input distribution. The result is
archived as a negative single-seed performance pilot, not as a basis for
target-set tuning.

## Artifacts

- `results/fwc-t2-colorfix-cuda-20260910-seed2026/metrics.json`
- `results/fwc-t2-colorfix-cuda-20260910-seed2026/status.json`
- `results/fwc-t2-colorfix-cuda-20260910-seed2026/protocol_audit.json`
- `checkpoints/fwc-t2-colorfix-cuda-20260910-seed2026/`

## Next step

Use the corrected run-time manifest generation, then rerun this requested T2
configuration once under CUDA. Do not launch a five-seed FWC comparison until
the audit reports `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04` as source
conditions and `N15_M07_F10` as target. The source-only contribution audit and
label/protocol semantics must also be verified first.
