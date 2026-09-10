# Corrected PU T2 FWC pilot

## Experiment goal

Verify that the corrected runtime manifest honors the requested PU transfer
task `[[0, 1, 2], 3]`, then measure one CUDA FWC run without changing the
model, loss, optimizer, or seed policy.

## Code and environment

- repo: `pangyu5451/AEI`
- local path: `D:\pycharm\AEI\DG-for-RMFD-main\AGG_FWC`
- code change: runtime manifest generation from `transfer_task`
- Python: `3.10.9`
- PyTorch: `2.11.0+cu128`
- CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 3080 Ti`
- seed: `2026`
- epoch: `60`
- CUDA fallback: `false`

## Exact command

```powershell
& 'D:\pycharm\robust_dg_fault\.venv\Scripts\python.exe' -m AGG_FWC.run --data_name PU --transfer_task "[[0,1,2],3]" --protocol-mode strict --fwc-stage fwc --require-cuda --seed 2026 --run_id fwc-t2-colorfix-mapped-cuda-20260910-seed2026 --data_root D:\datasets\PU --output_root D:\pycharm\AEI\DG-for-RMFD-main\AGG_FWC
```

## Protocol verification

The generated audit reports:

- source: `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`
- target: `N15_M07_F10`
- protocol mode: `paper_condition`
- bearing disjoint: `false`
- target evaluation count: `1`

The task is therefore a condition-transfer bearing-identity classification
run under the current manifest semantics, not unseen-bearing physical-fault
generalization. The manifest labels remain
`fault_label=bearing_class:<bearing_id>`.

## Results

| Metric | Result |
|---|---:|
| Accuracy | 0.317857 |
| Macro-F1 | 0.218568 |
| Weighted-F1 | 0.218568 |
| Target evaluation count | 1 |
| CUDA fallback | false |

This is a single-seed feasibility result. It is not compared directly with the
previous pilot because that run accidentally evaluated N09 as target despite
being requested as `[[0,1,2],3]`; the corrected run evaluates N15_M07_F10.

## Interpretation

The protocol bug is fixed and the run completed on CUDA. The color-direction
correction plus corrected T2 mapping does not provide evidence of a performance
improvement from this single seed. The result is archived as a negative
feasibility pilot; no target-set selection or tuning was performed.

## Artifacts

- `results/fwc-t2-colorfix-mapped-cuda-20260910-seed2026/config.json`
- `results/fwc-t2-colorfix-mapped-cuda-20260910-seed2026/metrics.json`
- `results/fwc-t2-colorfix-mapped-cuda-20260910-seed2026/status.json`
- `results/fwc-t2-colorfix-mapped-cuda-20260910-seed2026/protocol_audit.json`
- `results/fwc-t2-colorfix-mapped-cuda-20260910-seed2026/pu_condition_manifest.csv`

All run artifacts remain local and are excluded from version control.
