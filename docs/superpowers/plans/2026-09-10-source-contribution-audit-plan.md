# Source-Only Feature Contribution Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a source-only, record-level feature masking audit that measures NLL and Macro-F1 changes for frozen AGG features without changing training or FWC packing.

**Architecture:** A new `feature_contribution_audit` module validates source windows, runs a frozen classifier in evaluation mode, aggregates probabilities by condition and record, and computes global/condition-level metric deltas. A numeric NPZ plus JSON sidecar stores the report. Existing training and FWC modules remain unchanged except for the already-approved color-direction correction.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, scikit-learn, pytest.

---

### Task 1: Add source-only audit data contract and metric tests

**Files:**
- Create: `models/feature_contribution_audit.py`
- Create: `tests/test_feature_contribution_audit.py`

- [ ] **Step 1: Write the failing tests**

Add a tiny deterministic linear classifier and source windows with two records in two conditions. Assert that record aggregation is used, an informative feature has a larger positive NLL/F1 effect than a noise feature, condition order is sorted, and model parameters/training state are preserved.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_feature_contribution_audit.py -q
```

Expected: collection or import failure because `models.feature_contribution_audit` does not yet exist.

- [ ] **Step 3: Implement the minimal audit**

Implement `FeatureContributionAudit` and `audit_source_feature_contributions` with the exact contract in the design document. Use source median replacement by default, batched forward passes under `torch.no_grad()`, composite `(condition_id, record_id)` grouping, record-level probability means, record-level NLL, and Macro-F1. Snapshot model state tensors, restore the original training flag in `finally`, and raise if parameters or buffers changed.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run the same pytest command and require all tests in the new file to pass.

- [ ] **Step 5: Commit**

```powershell
git add models/feature_contribution_audit.py tests/test_feature_contribution_audit.py
git commit -m "feat: add source-only feature contribution audit"
```

### Task 2: Add safe persistence for audit reports

**Files:**
- Modify: `models/feature_contribution_audit.py`
- Modify: `tests/test_feature_contribution_audit.py`

- [ ] **Step 1: Write failing persistence tests**

Test `save_feature_contribution_audit` and `load_feature_contribution_audit` round-trip all arrays and metadata, reject overwrite, reject object arrays, and reject missing/false `source_only` metadata.

- [ ] **Step 2: Run the persistence tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_feature_contribution_audit.py -q
```

Expected: failure because persistence functions are absent.

- [ ] **Step 3: Implement persistence**

Write arrays to `<path>.npz` with `allow_pickle=False` compatibility and metadata to `<path>.json`. Refuse to overwrite either output. Validate schema version, source-only marker, dimensions, finite values, and condition count when loading.

- [ ] **Step 4: Run focused tests**

Run the same command and require all tests to pass.

- [ ] **Step 5: Commit**

```powershell
git add models/feature_contribution_audit.py tests/test_feature_contribution_audit.py
git commit -m "feat: persist source-only contribution audit reports"
```

### Task 3: Add public exports and documentation

**Files:**
- Modify: `models/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_feature_contribution_audit.py`

- [ ] **Step 1: Write export/documentation tests**

Assert the public functions can be imported from `AGG_FWC.models` and that the README includes the source-only/no-target boundary and report file contract.

- [ ] **Step 2: Implement the exports and README section**

Expose the audit types/functions and add a concise usage example showing source features, labels, record IDs, condition IDs, a frozen classifier, and report output. Explicitly state that the report is diagnostic and is not used to tune target performance.

- [ ] **Step 3: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_feature_contribution_audit.py -q
python -m pytest -q
```

Expected: new tests pass and the existing suite remains green.

- [ ] **Step 4: Commit**

```powershell
git add models/__init__.py README.md tests/test_feature_contribution_audit.py
git commit -m "docs: expose source-only contribution audit"
```

### Task 4: Run isolated audit pilot and archive evidence

**Files:**
- Create: `docs/experiments/2026-09-10-source-contribution-audit-pilot.md`
- Create: `scripts/run_feature_contribution_audit_pilot.py`

- [ ] **Step 1: Add a deterministic synthetic pilot**

Use the same tiny frozen classifier as the test, run with seed `2026`, save an audit report under a new run-specific directory, and print the top five features by NLL increase. Do not read PU/HUST target data.

- [ ] **Step 2: Run the pilot**

Run:

```powershell
python scripts/run_feature_contribution_audit_pilot.py --output_root runs --run_id contribution-audit-pilot-20260910
```

Expected: one numeric NPZ, one JSON metadata file, and a Markdown evidence record with command, seed, environment, and top-feature summary.

- [ ] **Step 3: Verify artifact isolation**

Confirm the new run does not overwrite any existing result directory and that no target path or model checkpoint is read.

- [ ] **Step 4: Commit the pilot runner and evidence**

```powershell
git add scripts/run_feature_contribution_audit_pilot.py docs/experiments/2026-09-10-source-contribution-audit-pilot.md
git commit -m "experiment: archive source contribution audit pilot"
```

### Task 5: Verify the corrected FWC color direction separately

**Files:**
- Existing: `models/feature_combination.py`
- Existing: `tests/test_feature_combination.py`
- Create: `docs/experiments/2026-09-10-fwc-color-direction.md`

- [ ] **Step 1: Run the focused and full verification**

```powershell
python -m pytest tests/test_feature_combination.py -q
python -m pytest -q
```

- [ ] **Step 2: Record the rule correction**

Document that only the contribution ranking direction changed: low contribution maps toward gray and high contribution maps toward red, while quotas, item values, red eligibility, and packing remain unchanged.

- [ ] **Step 3: Commit the experiment note**

```powershell
git add docs/experiments/2026-09-10-fwc-color-direction.md
git commit -m "docs: record FWC color direction correction"
```

