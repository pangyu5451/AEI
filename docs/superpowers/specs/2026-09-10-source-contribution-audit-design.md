# Source-Only Feature Contribution Audit Design

## Goal

Add a reproducible, source-only audit that measures whether each frozen AGG feature dimension has a defensible contribution to record-level classification before its value is used by FWC. The audit is diagnostic only: it must not change model weights, training losses, packing rules, or target evaluation.

## Scope

The audit accepts a frozen PyTorch classifier and already extracted feature windows. It replaces one feature at a time with a source-training reference value, evaluates the original and masked windows, aggregates window probabilities by `record_id`, and reports record-level changes in NLL and Macro-F1. It repeats the calculation for each source condition and seed supplied by the caller.

The implementation must:

- accept only source arrays; there is no target-data argument or target-data path;
- require one condition ID, record ID, and integer label per window;
- aggregate probabilities by record before computing metrics;
- use a frozen model in evaluation mode and never call backward or mutate parameters;
- use a deterministic feature reference vector, defaulting to the source-training median;
- preserve feature order so output index `j` always means extractor dimension `j`;
- return numeric arrays that can be saved with `allow_pickle=False`;
- record the model/device/seed/reference policy in the JSON metadata;
- keep explanation output separate from `assign_items`, `FWCFeatureCombination`, and training.

## Non-goals

This change does not:

- alter AGG, the extractor, classifier, FWC attributes, item colors, prices, collectible rules, or 3x3 packing;
- select a checkpoint using target performance;
- compute SHAP values or claim causal feature importance;
- replace source validation or provide a strict bearing-disjoint PU manifest;
- make the existing full experiment runner automatically invoke the audit.

## Data contract

The public function receives:

```python
audit_source_feature_contributions(
    model,
    features,          # [N, D] float array or CPU/GPU tensor
    labels,            # [N] integer class labels
    record_ids,        # [N] non-empty group identifiers
    condition_ids,     # [N] non-empty source-condition identifiers
    *,
    device,
    seed=2026,
    reference=None,
    batch_size=64,
)
```

The function returns an immutable-style mapping containing:

- `feature_indices`: `[D]` integer indices;
- `global_nll_increase`: `[D]`, masked record-level NLL minus baseline NLL;
- `global_macro_f1_drop`: `[D]`, baseline record Macro-F1 minus masked Macro-F1;
- `condition_nll_increase`: `[K, D]` in sorted condition order;
- `condition_macro_f1_drop`: `[K, D]` in the same order;
- `record_count_by_condition`: `[K]`;
- `reference`: `[D]` source reference values;
- `condition_ids`: ordered condition names;
- `seed`, `batch_size`, `feature_count`, and `model_training_was_disabled`.

The audit must reject empty arrays, non-finite features, non-integer labels, missing IDs, mismatched lengths, fewer than two records, fewer than two classes, and a reference vector whose length is not `D`. It must reject a model that remains in training mode after the function enters evaluation, and restore the original training/evaluation state before returning.

## Algorithm

1. Validate all arrays and normalize condition/record IDs to stable strings for grouping only.
2. Compute the source median reference vector if no reference is supplied. The reference is calculated from the passed source features only.
3. Put a copy of the model in evaluation mode on the requested device without gradients. Do not alter parameter values.
4. Compute baseline window probabilities in batches.
5. For each feature index, replace that column with the fixed reference value, compute masked probabilities in batches, and aggregate both baseline and masked probabilities by `(condition_id, record_id)` using probability means.
6. For every condition and globally, compute record-level NLL using the true record label and Macro-F1 from the argmax prediction. A positive NLL increase or Macro-F1 drop means masking the feature harmed classification.
7. Restore model training state and return only detached numeric results.

The implementation may use CPU NumPy/scikit-learn for final scalar metrics because this is a post-hoc source-only report, but model forward passes and masking must remain on the selected device. No target tensor may be created or read.

## Persistence

Add `save_feature_contribution_audit(report, path)` and `load_feature_contribution_audit(path)` using a compressed NPZ for arrays and a sidecar JSON for metadata. Both writers must refuse to overwrite an existing file. The saved report must include a schema version and source-only marker. Loading must reject missing arrays, object arrays, non-finite values, inconsistent dimensions, or a false/missing source-only marker.

## Testing

Tests must cover:

1. record aggregation changes the metric unit from windows to records;
2. masking a deliberately informative feature produces a larger positive NLL increase and Macro-F1 drop than masking noise;
3. condition-level output has stable sorted order and correct shapes;
4. supplied reference values are honored and median reference is deterministic;
5. model parameters and original training mode are unchanged;
6. malformed arrays, IDs, labels, reference vectors, and non-finite values are rejected;
7. save/load round trip uses only numeric NPZ arrays and preserves metadata;
8. no function signature contains a target-data argument.

## Acceptance criteria

- New tests fail before implementation and pass after implementation.
- Existing full test suite remains green.
- The audit can run on a small synthetic frozen classifier without a dataset or checkpoint.
- No existing training or FWC inference test changes its expected behavior.
- The first experiment produces a source-only report; it does not modify or overwrite existing run directories.

