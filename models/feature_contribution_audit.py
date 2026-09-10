"""Source-only, record-level feature contribution auditing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch


_SCHEMA_VERSION = 1
_MIN_PROBABILITY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class FeatureContributionAudit:
    """Numeric source-only feature masking report."""

    feature_indices: tuple[int, ...]
    global_nll_increase: tuple[float, ...]
    global_macro_f1_drop: tuple[float, ...]
    condition_ids: tuple[str, ...]
    condition_nll_increase: tuple[tuple[float, ...], ...]
    condition_macro_f1_drop: tuple[tuple[float, ...], ...]
    record_count_by_condition: tuple[int, ...]
    reference: tuple[float, ...]
    seed: int
    batch_size: int
    feature_count: int
    model_training_was_disabled: bool = True
    schema_version: int = _SCHEMA_VERSION
    source_only: bool = True

    def __post_init__(self):
        feature_indices = tuple(int(value) for value in self.feature_indices)
        global_nll = tuple(float(value) for value in self.global_nll_increase)
        global_f1 = tuple(float(value) for value in self.global_macro_f1_drop)
        condition_ids = tuple(str(value) for value in self.condition_ids)
        condition_nll = tuple(tuple(float(value) for value in row) for row in self.condition_nll_increase)
        condition_f1 = tuple(tuple(float(value) for value in row) for row in self.condition_macro_f1_drop)
        counts = tuple(int(value) for value in self.record_count_by_condition)
        reference = tuple(float(value) for value in self.reference)
        if self.schema_version != _SCHEMA_VERSION or self.source_only is not True:
            raise ValueError("audit must use schema version 1 and source_only=true")
        if self.feature_count <= 0 or len(feature_indices) != self.feature_count:
            raise ValueError("feature_indices must match feature_count")
        if feature_indices != tuple(range(self.feature_count)):
            raise ValueError("feature_indices must be consecutive from zero")
        if len(global_nll) != self.feature_count or len(global_f1) != self.feature_count:
            raise ValueError("global feature metrics must match feature_count")
        if len(reference) != self.feature_count:
            raise ValueError("reference must match feature_count")
        if not condition_ids or len(condition_nll) != len(condition_ids) or len(condition_f1) != len(condition_ids):
            raise ValueError("condition metrics must match condition_ids")
        if len(counts) != len(condition_ids) or any(value <= 0 for value in counts):
            raise ValueError("record counts must match non-empty conditions")
        for rows in (condition_nll, condition_f1):
            if any(len(row) != self.feature_count for row in rows):
                raise ValueError("condition metrics must match feature_count")
        all_values = [*global_nll, *global_f1, *reference]
        all_values.extend(value for row in condition_nll for value in row)
        all_values.extend(value for row in condition_f1 for value in row)
        if any(not np.isfinite(value) for value in all_values):
            raise ValueError("audit values must be finite")
        if any(value < 0 for value in counts):
            raise ValueError("record counts must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, (int, np.integer)) or self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        object.__setattr__(self, "feature_indices", feature_indices)
        object.__setattr__(self, "global_nll_increase", global_nll)
        object.__setattr__(self, "global_macro_f1_drop", global_f1)
        object.__setattr__(self, "condition_ids", condition_ids)
        object.__setattr__(self, "condition_nll_increase", condition_nll)
        object.__setattr__(self, "condition_macro_f1_drop", condition_f1)
        object.__setattr__(self, "record_count_by_condition", counts)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "feature_count", int(self.feature_count))


def audit_source_feature_contributions(
    model,
    features,
    labels,
    record_ids,
    condition_ids,
    *,
    device,
    seed=2026,
    reference=None,
    batch_size=64,
):
    """Measure source-only record-level effects of masking each feature."""

    values, label_values, record_values, condition_values = _validate_inputs(
        features, labels, record_ids, condition_ids
    )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not hasattr(model, "eval") or not callable(model):
        raise ValueError("model must be a callable PyTorch module")
    reference_values = (
        np.median(values, axis=0) if reference is None else _validate_reference(reference, values.shape[1])
    )
    groups = _record_groups(condition_values, record_values, label_values)
    working_model = deepcopy(model)
    working_model.to(torch.device(device))
    working_model.eval()
    with torch.no_grad():
        baseline_probabilities = _forward_probabilities(
            working_model, values, device=device, batch_size=int(batch_size)
        )
    baseline_metrics = _record_metrics(baseline_probabilities, groups)
    feature_count = values.shape[1]
    global_nll = np.empty(feature_count, dtype=float)
    global_f1 = np.empty(feature_count, dtype=float)
    condition_nll = np.empty((len(baseline_metrics["condition_ids"]), feature_count), dtype=float)
    condition_f1 = np.empty_like(condition_nll)
    for feature_index in range(feature_count):
        masked = values.copy()
        masked[:, feature_index] = reference_values[feature_index]
        with torch.no_grad():
            masked_probabilities = _forward_probabilities(
                working_model, masked, device=device, batch_size=int(batch_size)
            )
        masked_metrics = _record_metrics(masked_probabilities, groups)
        global_nll[feature_index] = masked_metrics["global_nll"] - baseline_metrics["global_nll"]
        global_f1[feature_index] = baseline_metrics["global_macro_f1"] - masked_metrics["global_macro_f1"]
        condition_nll[:, feature_index] = (
            masked_metrics["condition_nll"] - baseline_metrics["condition_nll"]
        )
        condition_f1[:, feature_index] = (
            baseline_metrics["condition_macro_f1"] - masked_metrics["condition_macro_f1"]
        )
    return FeatureContributionAudit(
        feature_indices=tuple(range(feature_count)),
        global_nll_increase=tuple(global_nll.tolist()),
        global_macro_f1_drop=tuple(global_f1.tolist()),
        condition_ids=baseline_metrics["condition_ids"],
        condition_nll_increase=tuple(tuple(row) for row in condition_nll.tolist()),
        condition_macro_f1_drop=tuple(tuple(row) for row in condition_f1.tolist()),
        record_count_by_condition=baseline_metrics["record_counts"],
        reference=tuple(reference_values.tolist()),
        seed=int(seed),
        batch_size=int(batch_size),
        feature_count=feature_count,
    )


def save_feature_contribution_audit(report, path):
    """Save a report as numeric NPZ plus JSON metadata without overwriting."""

    if not isinstance(report, FeatureContributionAudit):
        raise ValueError("report must be a FeatureContributionAudit")
    npz_path, json_path = _report_paths(path)
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("audit report already exists")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        feature_indices=np.asarray(report.feature_indices, dtype=np.int64),
        global_nll_increase=np.asarray(report.global_nll_increase, dtype=float),
        global_macro_f1_drop=np.asarray(report.global_macro_f1_drop, dtype=float),
        condition_nll_increase=np.asarray(report.condition_nll_increase, dtype=float),
        condition_macro_f1_drop=np.asarray(report.condition_macro_f1_drop, dtype=float),
        record_count_by_condition=np.asarray(report.record_count_by_condition, dtype=np.int64),
        reference=np.asarray(report.reference, dtype=float),
    )
    metadata = {
        "schema_version": report.schema_version,
        "source_only": report.source_only,
        "condition_ids": list(report.condition_ids),
        "seed": report.seed,
        "batch_size": report.batch_size,
        "feature_count": report.feature_count,
        "model_training_was_disabled": report.model_training_was_disabled,
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path


def load_feature_contribution_audit(path):
    """Load and validate a numeric feature contribution audit."""

    npz_path, json_path = _report_paths(path)
    if not npz_path.exists() or not json_path.exists():
        raise ValueError("audit report requires both NPZ and JSON files")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != _SCHEMA_VERSION or metadata.get("source_only") is not True:
        raise ValueError("audit metadata must identify schema version 1 and source_only=true")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "feature_indices",
            "global_nll_increase",
            "global_macro_f1_drop",
            "condition_nll_increase",
            "condition_macro_f1_drop",
            "record_count_by_condition",
            "reference",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("audit report is missing arrays: " + ", ".join(sorted(missing)))
        arrays = {name: np.asarray(archive[name]) for name in required}
    if any(values.dtype.kind not in "biuf" for values in arrays.values()):
        raise ValueError("audit arrays must be numeric")
    feature_count = int(metadata.get("feature_count", 0))
    condition_ids = tuple(metadata.get("condition_ids", ()))
    return FeatureContributionAudit(
        feature_indices=tuple(arrays["feature_indices"].tolist()),
        global_nll_increase=tuple(arrays["global_nll_increase"].tolist()),
        global_macro_f1_drop=tuple(arrays["global_macro_f1_drop"].tolist()),
        condition_ids=condition_ids,
        condition_nll_increase=tuple(tuple(row) for row in arrays["condition_nll_increase"].tolist()),
        condition_macro_f1_drop=tuple(tuple(row) for row in arrays["condition_macro_f1_drop"].tolist()),
        record_count_by_condition=tuple(arrays["record_count_by_condition"].tolist()),
        reference=tuple(arrays["reference"].tolist()),
        seed=metadata.get("seed"),
        batch_size=metadata.get("batch_size"),
        feature_count=feature_count,
        model_training_was_disabled=metadata.get("model_training_was_disabled") is True,
    )


def _forward_probabilities(model, values, *, device, batch_size):
    tensor = torch.as_tensor(values, dtype=torch.float32, device=torch.device(device))
    outputs = []
    for start in range(0, tensor.shape[0], batch_size):
        logits = model(tensor[start : start + batch_size])
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("model must return a two-dimensional logits tensor")
        outputs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    probabilities = np.concatenate(outputs, axis=0)
    if not np.isfinite(probabilities).all() or probabilities.shape[1] < 2:
        raise ValueError("model probabilities must be finite with at least two classes")
    return probabilities


def _record_metrics(probabilities, groups):
    record_probabilities = []
    record_labels = []
    record_conditions = []
    for condition_id, record_id, indices, label in groups:
        del record_id
        record_probabilities.append(probabilities[indices].mean(axis=0))
        record_labels.append(label)
        record_conditions.append(condition_id)
    record_probabilities = np.asarray(record_probabilities, dtype=float)
    record_labels = np.asarray(record_labels, dtype=np.int64)
    record_conditions = np.asarray(record_conditions, dtype=str)
    if (record_labels < 0).any() or (record_labels >= record_probabilities.shape[1]).any():
        raise ValueError("labels must fit the model class range")
    predictions = record_probabilities.argmax(axis=1)
    classes = np.unique(record_labels)
    global_nll = float(-np.log(np.clip(record_probabilities[np.arange(len(record_labels)), record_labels], _MIN_PROBABILITY, 1.0)).mean())
    global_macro_f1 = float(
        f1_score(record_labels, predictions, labels=classes, average="macro", zero_division=0)
    )
    condition_ids = tuple(sorted(set(record_conditions.tolist())))
    condition_nll = []
    condition_f1 = []
    record_counts = []
    for condition_id in condition_ids:
        selected = np.flatnonzero(record_conditions == condition_id)
        selected_labels = record_labels[selected]
        selected_probabilities = record_probabilities[selected]
        selected_predictions = predictions[selected]
        selected_classes = np.unique(selected_labels)
        condition_nll.append(
            float(-np.log(np.clip(selected_probabilities[np.arange(len(selected)), selected_labels], _MIN_PROBABILITY, 1.0)).mean())
        )
        condition_f1.append(
            float(f1_score(selected_labels, selected_predictions, labels=selected_classes, average="macro", zero_division=0))
        )
        record_counts.append(int(selected.size))
    return {
        "global_nll": global_nll,
        "global_macro_f1": global_macro_f1,
        "condition_ids": condition_ids,
        "condition_nll": np.asarray(condition_nll, dtype=float),
        "condition_macro_f1": np.asarray(condition_f1, dtype=float),
        "record_counts": tuple(record_counts),
    }


def _record_groups(condition_ids, record_ids, labels):
    keys = sorted(set(zip(condition_ids.tolist(), record_ids.tolist())))
    groups = []
    for condition_id, record_id in keys:
        indices = np.flatnonzero((condition_ids == condition_id) & (record_ids == record_id))
        unique_labels = np.unique(labels[indices])
        if unique_labels.size != 1:
            raise ValueError("each condition/record group must have one label")
        groups.append((condition_id, record_id, indices, int(unique_labels[0])))
    return tuple(groups)


def _validate_inputs(features, labels, record_ids, condition_ids):
    if isinstance(features, torch.Tensor):
        values = features.detach().cpu().numpy()
    else:
        values = np.asarray(features)
    try:
        values = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("features must be numeric") from error
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    label_values = np.asarray(labels)
    if label_values.ndim != 1 or not np.issubdtype(label_values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    record_values = _normalize_ids(record_ids, "record IDs")
    condition_values = _normalize_ids(condition_ids, "condition IDs")
    if not (len(label_values) == len(record_values) == len(condition_values) == values.shape[0]):
        raise ValueError("features, labels, record IDs, and condition IDs must have the same length")
    if np.unique(label_values).size < 2:
        raise ValueError("labels must contain at least two classes")
    if len(set(zip(condition_values.tolist(), record_values.tolist()))) < 2:
        raise ValueError("source data must contain at least two records")
    return values, label_values.astype(np.int64), record_values, condition_values


def _normalize_ids(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    normalized = []
    for value in array.tolist():
        text = str(value)
        if text.strip() in {"", "None", "nan"}:
            raise ValueError(f"{name} must contain non-empty values")
        normalized.append(text)
    return np.asarray(normalized, dtype=str)


def _validate_reference(reference, feature_count):
    values = np.asarray(reference, dtype=np.float32)
    if values.ndim != 1 or values.size != feature_count:
        raise ValueError("reference must be a one-dimensional vector matching feature count")
    if not np.isfinite(values).all():
        raise ValueError("reference must be finite")
    return values


def _report_paths(path):
    base = Path(path)
    if base.suffix.lower() in {".npz", ".json"}:
        base = base.with_suffix("")
    return base.with_suffix(".npz"), base.with_suffix(".json")

