# Source-only feature contribution audit pilot

## Experiment goal

Verify the source-only masking audit, record-level probability aggregation,
numeric report persistence, and deterministic top-feature ordering before using
the audit on a frozen AGG checkpoint. This is a software pilot, not a PU
performance result.

## Code version

- repo: `pangyu5451/AEI`
- local path: `D:\pycharm\AEI\DG-for-RMFD-main\AGG_FWC`
- commit: recorded in `runs/contribution-audit-pilot-20260910/run_metadata.json`
- branch: `master`

## Environment

- Python: recorded in `run_metadata.json`
- PyTorch/CUDA: recorded in `run_metadata.json`
- device: `cpu` (synthetic pilot)

## Data

- dataset: deterministic synthetic source-only data
- records: 4, with 3 windows per record
- conditions: `a`, `z`
- classes: 2
- target data: not supplied or read
- preprocessing: feature 0 is informative; feature 1 is noise; default masking reference is the source median

## Parameters

- seed: `2026`
- batch size: `4`
- masking unit: one feature column at a time
- metric unit: record-level mean probability, not window-level output

## Results

| Metric | Result |
|---|---:|
| Source-only marker | `true` |
| Feature count | `2` |
| Records per condition | `a=2`, `z=2` |
| Top feature by global NLL increase | feature `0` |
| Feature 0 NLL increase | `0.2864450179` |
| Feature 0 Macro-F1 drop | `0.6666666667` |
| Feature 1 NLL increase | `0.0000000000` |
| Feature 1 Macro-F1 drop | `0.0000000000` |

## Observation

The audit correctly identifies the deliberately informative feature and keeps
the noise feature at zero effect. Condition order is deterministic (`a`, `z`),
the model is evaluated without changing the caller's training state or
parameters, and the saved NPZ contains numeric arrays only.

## Artifacts

- `runs/contribution-audit-pilot-20260910/feature_contribution_audit.npz`
- `runs/contribution-audit-pilot-20260910/feature_contribution_audit.json`
- `runs/contribution-audit-pilot-20260910/summary.json`
- `runs/contribution-audit-pilot-20260910/run_metadata.json`

## Next step

Run the same audit on source-train/source-validation features extracted from a
frozen AGG checkpoint, after the physical-label/protocol semantics are fixed.
Do not use the audit to inspect or tune the target set.

