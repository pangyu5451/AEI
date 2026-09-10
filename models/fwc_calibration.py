"""Source-only, record-grouped FWC attribute calibration helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from AGG_FWC.models.fwc_types import (
    ConditionAttributeReport,
    _create_audited_source_payload,
    _require_audited_source_partition,
)


_MAX_RESAMPLE_ATTEMPTS = 64


def create_audited_source_payload(
    audited_source_partition,
    source_features,
    source_condition_ids,
    source_record_ids,
    attribute_reports,
):
    """Attest one payload against an audited partition in the trusted-runner model.

    This catches accidental source/target mixing.  It is not hostile-process
    security and does not replace Task7 target-loader enforcement.
    """
    _require_audited_source_partition(audited_source_partition)
    record_ids = tuple(str(record_id) for record_id in source_record_ids)
    condition_ids = tuple(str(condition_id) for condition_id in source_condition_ids)
    if not record_ids or len(record_ids) != len(condition_ids):
        raise ValueError("source record IDs and condition IDs must have matching non-empty length")
    if not set(record_ids).issubset(audited_source_partition.source_record_ids):
        raise ValueError("audited partition rejects non-source record IDs")
    if set(record_ids).intersection(audited_source_partition.target_record_ids):
        raise ValueError("audited partition rejects declared target record IDs")
    if not set(condition_ids).issubset(audited_source_partition.source_condition_ids):
        raise ValueError("audited partition rejects non-source condition IDs")
    report_ids = set()
    for report in attribute_reports:
        if not isinstance(report, ConditionAttributeReport):
            raise ValueError("attribute_reports must contain ConditionAttributeReport values")
        report_ids.add(report.condition_id)
    if not report_ids or not report_ids.issubset(audited_source_partition.source_condition_ids):
        raise ValueError("audited partition rejects non-source attribute reports")
    return _create_audited_source_payload(
        attribute_training_fingerprint(
            source_features, condition_ids, record_ids, attribute_reports
        )
    )


def attribute_training_fingerprint(
    source_features, source_condition_ids, source_record_ids, attribute_reports
):
    """Return a deterministic digest for one feature/report training payload."""
    features = _feature_array(source_features)
    reports = []
    for report in attribute_reports:
        if not isinstance(report, ConditionAttributeReport):
            raise ValueError("attribute_reports must contain ConditionAttributeReport values")
        reports.append(
            {
                "condition_id": report.condition_id,
                "contribution": list(report.contribution),
                "stability": list(report.stability),
                "redundancy": list(report.redundancy),
                "c_quantiles": list(report.c_quantiles),
                "delta_margin_quantiles": list(report.delta_margin_quantiles),
            }
        )
    payload = {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "condition_ids": list(source_condition_ids),
        "record_ids": list(source_record_ids),
        "reports": reports,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _feature_array(features):
    if hasattr(features, "detach"):
        features = features.detach().cpu().contiguous().numpy()
    values = np.asarray(features)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("source features must be a finite two-dimensional array")
    return np.ascontiguousarray(values)


def utility(c, s, r):
    """Return the clipped data-derived FWC utility ``C * S * (1 - R)``."""
    return np.clip(c, 0.0, 1.0) * np.clip(s, 0.0, 1.0) * (1.0 - np.clip(r, 0.0, 1.0))


def calibrate_condition(
    features, labels, record_ids, repeats=5, seed=2026, *, condition_id="source"
):
    """Calibrate contribution, stability, and redundancy from source records only."""
    features, labels, record_ids = _validate_source_inputs(features, labels, record_ids)
    _validate_repeats_and_seed(repeats, seed)
    condition_id = _validate_condition_id(condition_id)

    raw_contribution, delta_margins = _feature_contribution(features, labels, record_ids, seed)
    contribution = _normalize_unit_interval(raw_contribution)
    resampled_contributions = []
    for resample_index in range(5):
        resampled_features, resampled_labels, resampled_ids = _resample_groups(
            features, labels, record_ids, seed + resample_index
        )
        values, _ = _feature_contribution(
            resampled_features, resampled_labels, resampled_ids, seed + resample_index
        )
        resampled_contributions.append(values)

    stability = _inverse_normalized_std(np.std(resampled_contributions, axis=0))
    redundancy = _redundancy(features, contribution)
    return ConditionAttributeReport(
        condition_id=condition_id,
        contribution=contribution,
        stability=stability,
        redundancy=redundancy,
        c_quantiles=_quantiles(contribution),
        delta_margin_quantiles=_quantiles(delta_margins),
    )


def single_feature_grouped_oof_accuracies(features, labels, record_ids, seed=2026):
    """Return source-only grouped OOF accuracies for one regularized model per feature."""
    features, labels, record_ids = _validate_source_inputs(features, labels, record_ids)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    accuracies = np.empty(features.shape[1], dtype=float)
    for feature_index in range(features.shape[1]):
        predictions, _ = _grouped_oof_predictions(
            features[:, [feature_index]], labels, record_ids, int(seed) + feature_index
        )
        accuracies[feature_index] = np.mean(predictions == labels)
    return accuracies


def select_external_features(condition_accuracies):
    """Select features meeting both source-condition accuracy thresholds."""
    values = np.asarray(condition_accuracies, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("condition accuracies must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("condition accuracies must be finite")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("condition accuracies must be within [0, 1]")
    return (values.mean(axis=0) >= 0.85) & (values.max(axis=0) >= 0.95)


def save_condition_attribute_report(report, path):
    """Persist a report with numeric, ``allow_pickle=False``-readable NPZ arrays."""
    if not isinstance(report, ConditionAttributeReport):
        raise ValueError("report must be a ConditionAttributeReport")
    arrays = _report_arrays(report)
    output_path = Path(path)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_condition_attribute_report(path):
    """Load a condition calibration report written by ``save_condition_attribute_report``."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "condition_id",
            "contribution",
            "stability",
            "redundancy",
            "c_quantiles",
            "delta_margin_quantiles",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("report is missing arrays: " + ", ".join(sorted(missing)))
        condition_id = str(np.asarray(archive["condition_id"]).item())
        contribution = np.asarray(archive["contribution"], dtype=float)
        stability = np.asarray(archive["stability"], dtype=float)
        redundancy = np.asarray(archive["redundancy"], dtype=float)
        c_quantiles = tuple(np.asarray(archive["c_quantiles"], dtype=float).tolist())
        delta_margin_quantiles = tuple(
            np.asarray(archive["delta_margin_quantiles"], dtype=float).tolist()
        )
    return _validated_report(
        condition_id,
        contribution,
        stability,
        redundancy,
        c_quantiles,
        delta_margin_quantiles,
    )


def _feature_contribution(features, labels, record_ids, seed):
    baseline_predictions, baseline_probabilities = _grouped_oof_predictions(
        features, labels, record_ids, seed
    )
    classes = np.unique(labels)
    baseline_score = f1_score(labels, baseline_predictions, labels=classes, average="macro", zero_division=0)
    contribution = np.empty(features.shape[1], dtype=float)
    delta_margins = []
    for feature_index in range(features.shape[1]):
        masked = features.copy()
        masked[:, feature_index] = 0.0
        masked_predictions, masked_probabilities = _grouped_oof_predictions(
            masked, labels, record_ids, seed + feature_index + 1
        )
        masked_score = f1_score(labels, masked_predictions, labels=classes, average="macro", zero_division=0)
        contribution[feature_index] = baseline_score - masked_score
        baseline_class_indices = np.searchsorted(classes, baseline_predictions)
        masked_baseline_confidence = masked_probabilities[
            np.arange(labels.size), baseline_class_indices
        ]
        delta_margins.extend(
            np.maximum(0.0, baseline_probabilities.max(axis=1) - masked_baseline_confidence)
        )
    return contribution, np.asarray(delta_margins, dtype=float)


def _grouped_oof_predictions(features, labels, record_ids, seed):
    classes = np.unique(labels)
    predictions = np.empty(labels.shape, dtype=labels.dtype)
    probabilities = np.zeros((labels.size, classes.size), dtype=float)
    _assert_grouped_oof_class_coverage(labels, record_ids)
    splitter = GroupKFold(n_splits=min(5, _unique_record_ids(record_ids).size))
    for train_indices, test_indices in splitter.split(features, labels, groups=record_ids):
        train_labels = labels[train_indices]
        classifier = LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", random_state=int(seed)
        )
        classifier.fit(features[train_indices], train_labels)
        predictions[test_indices] = classifier.predict(features[test_indices])
        fold_probabilities = classifier.predict_proba(features[test_indices])
        for class_index, class_label in enumerate(classifier.classes_):
            probabilities[test_indices, np.searchsorted(classes, class_label)] = fold_probabilities[:, class_index]
    return predictions, probabilities


def _resample_groups(features, labels, record_ids, seed):
    groups = _unique_record_ids(record_ids)
    count = min(groups.size - 1, max(2, int(np.ceil(groups.size * 0.8))))
    if count < 2:
        raise ValueError(
            "stability resampling requires at least three groups to omit one group"
        )
    rng = np.random.default_rng(seed)
    for _ in range(_MAX_RESAMPLE_ATTEMPTS):
        selected = rng.choice(groups, size=count, replace=False)
        selected_indices = np.flatnonzero(np.isin(record_ids, selected))
        candidate_features = features[selected_indices]
        candidate_labels = labels[selected_indices]
        candidate_ids = record_ids[selected_indices]
        try:
            _assert_grouped_oof_class_coverage(candidate_labels, candidate_ids)
        except ValueError:
            continue
        return candidate_features, candidate_labels, candidate_ids
    raise ValueError(
        "no valid grouped OOF resample found after "
        f"{_MAX_RESAMPLE_ATTEMPTS} deterministic attempts"
    )


def _assert_grouped_oof_class_coverage(labels, record_ids):
    classes = np.unique(labels)
    splitter = GroupKFold(n_splits=min(5, _unique_record_ids(record_ids).size))
    placeholder_features = np.empty((labels.size, 1), dtype=float)
    for train_indices, _ in splitter.split(placeholder_features, labels, groups=record_ids):
        if np.unique(labels[train_indices]).size != classes.size:
            raise ValueError(
                "grouped-fold class coverage failure: each OOF training fold "
                "must contain every source class"
            )


def _inverse_normalized_std(std):
    values = np.asarray(std, dtype=float)
    lower, upper = values.min(), values.max()
    if np.isclose(lower, upper):
        return np.ones_like(values)
    return 1.0 - (values - lower) / (upper - lower)


def _normalize_unit_interval(values):
    values = np.asarray(values, dtype=float)
    lower, upper = values.min(), values.max()
    if np.isclose(lower, upper):
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def _redundancy(features, contribution):
    redundancy = np.zeros(features.shape[1], dtype=float)
    for index, value in enumerate(contribution):
        higher = np.flatnonzero(contribution > value)
        if higher.size:
            correlations = [abs(_pearson(features[:, index], features[:, other])) for other in higher]
            redundancy[index] = max(correlations)
    return redundancy


def _pearson(first, second):
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    correlation = float(np.corrcoef(first, second)[0, 1])
    return 0.0 if not np.isfinite(correlation) else correlation


def _quantiles(values):
    lower, upper = np.quantile(np.asarray(values, dtype=float), [0.05, 0.95])
    return float(lower), float(upper)


def _validate_source_inputs(features, labels, record_ids):
    try:
        feature_values = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("features must be numeric") from error
    label_values = np.asarray(labels)
    record_values = np.asarray(record_ids)
    if feature_values.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if label_values.ndim != 1 or record_values.ndim != 1:
        raise ValueError("labels and record IDs must be one-dimensional")
    if feature_values.shape[0] == 0 or feature_values.shape[1] == 0:
        raise ValueError("features must be non-empty")
    if feature_values.shape[0] != label_values.size or label_values.size != record_values.size:
        raise ValueError("features, labels, and record IDs must have the same length")
    if not np.isfinite(feature_values).all():
        raise ValueError("features must be finite")
    if any(value is None or (isinstance(value, str) and not value.strip()) for value in label_values):
        raise ValueError("labels must not contain missing values")
    if any(value is None or (isinstance(value, str) and not value.strip()) for value in record_values):
        raise ValueError("record IDs must not contain missing values")
    if np.unique(label_values).size < 2:
        raise ValueError("labels must contain at least two classes")
    if _unique_record_ids(record_values).size < 2:
        raise ValueError("record IDs must contain at least two groups")
    return feature_values, label_values, record_values


def _validate_repeats_and_seed(repeats, seed):
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats != 5:
        raise ValueError("repeats must be exactly five")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")


def _report_arrays(report):
    validated = _validated_report(
        report.condition_id,
        report.contribution,
        report.stability,
        report.redundancy,
        report.c_quantiles,
        report.delta_margin_quantiles,
    )
    return {
        "condition_id": np.asarray(validated.condition_id, dtype=np.str_),
        "contribution": np.asarray(validated.contribution, dtype=float),
        "stability": np.asarray(validated.stability, dtype=float),
        "redundancy": np.asarray(validated.redundancy, dtype=float),
        "c_quantiles": np.asarray(validated.c_quantiles, dtype=float),
        "delta_margin_quantiles": np.asarray(validated.delta_margin_quantiles, dtype=float),
    }


def _validated_report(condition_id, contribution, stability, redundancy, c_quantiles, delta_margin_quantiles):
    arrays = [np.asarray(values, dtype=float) for values in (contribution, stability, redundancy)]
    condition_id = _validate_condition_id(condition_id)
    if any(values.ndim != 1 for values in arrays) or not arrays[0].size:
        raise ValueError("report attributes must be non-empty one-dimensional arrays")
    if len({values.size for values in arrays}) != 1 or not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("report attributes must have matching finite values")
    quantiles = [tuple(values) for values in (c_quantiles, delta_margin_quantiles)]
    if any(len(values) != 2 or not np.isfinite(values).all() for values in quantiles):
        raise ValueError("report quantiles must contain two finite values")
    return ConditionAttributeReport(
        condition_id=condition_id,
        contribution=arrays[0],
        stability=arrays[1],
        redundancy=arrays[2],
        c_quantiles=quantiles[0],
        delta_margin_quantiles=quantiles[1],
    )


def _validate_condition_id(condition_id):
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError("condition_id must be a non-empty string")
    return condition_id


def _unique_record_ids(record_ids):
    try:
        return np.unique(record_ids)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "record IDs must be mutually comparable for uniqueness and grouped splits"
        ) from error
