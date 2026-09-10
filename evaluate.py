"""Evaluation and JSON serialization helpers for AGG_FWC runs."""

import json
import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def compute_classification_metrics(labels, predictions, num_classes):
    """Return JSON-ready classification metrics and a row=true confusion matrix."""
    if (
        not isinstance(num_classes, (int, np.integer))
        or isinstance(num_classes, (bool, np.bool_))
        or num_classes <= 0
    ):
        raise ValueError("num_classes must be a positive integer")
    num_classes = int(num_classes)
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    if labels_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("labels and predictions must be strictly one-dimensional")
    if labels_array.size != predictions_array.size:
        raise ValueError("labels and predictions must have the same length")
    if labels_array.size == 0:
        raise ValueError("labels and predictions must be non-empty")
    if not np.issubdtype(labels_array.dtype, np.integer) or not np.issubdtype(
        predictions_array.dtype, np.integer
    ):
        raise ValueError("labels and predictions must be valid class indices")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels_array, predictions_array):
        label = int(label)
        prediction = int(prediction)
        if not 0 <= label < num_classes or not 0 <= prediction < num_classes:
            raise ValueError("labels and predictions must be valid class indices")
        confusion[label, prediction] += 1

    total = int(labels_array.size)
    per_class = []
    f1_values = []
    supports = []
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        support = int(confusion[class_index, :].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
        f1_values.append(f1)
        supports.append(support)

    return {
        "accuracy": float(np.divide(confusion.diagonal().sum(), total)) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": (
            float(np.divide(np.dot(f1_values, supports), total)) if total else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_predictions_file(predictions_path, num_classes, output_path, metadata=None):
    """Recompute and save metrics from a saved ``predictions.npz`` file."""
    metadata = _validate_metadata(metadata)
    predictions_path = Path(predictions_path)
    with np.load(predictions_path, allow_pickle=False) as payload:
        required = {"labels", "predictions"}
        missing = required.difference(payload.files)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"predictions file must contain labels and predictions; missing {missing_names}")
        labels = payload["labels"]
        predictions = payload["predictions"]

    metrics = compute_classification_metrics(labels, predictions, num_classes)
    metrics.update(
        {
            "run_id": metadata["run_id"],
            "config_path": metadata["config_path"],
            "metadata": metadata,
        }
    )
    return save_metrics_json(metrics, output_path)


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must include run_id and config_path")
    required = ("run_id", "config_path")
    if any(key not in metadata for key in required):
        raise ValueError("metadata must include run_id and config_path")
    if not isinstance(metadata["run_id"], str) or not metadata["run_id"].strip():
        raise ValueError("metadata run_id must be a non-empty string")
    if not isinstance(metadata["config_path"], (str, Path)) or not str(
        metadata["config_path"]
    ).strip():
        raise ValueError("metadata config_path must be a non-empty path")
    return metadata


def _parse_metadata(value):
    def reject_non_finite(value):
        raise ValueError(f"non-finite JSON number {value}")

    try:
        metadata = json.loads(value, parse_constant=reject_non_finite)
    except (json.JSONDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "metadata must be valid JSON with finite numbers"
        ) from exc
    if not isinstance(metadata, dict):
        raise argparse.ArgumentTypeError("metadata must be a JSON object")
    try:
        _validate_metadata(metadata)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recompute classification metrics from predictions.npz"
    )
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metadata", type=_parse_metadata, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return evaluate_predictions_file(
        args.predictions_path,
        num_classes=args.num_classes,
        output_path=args.output_path,
        metadata=args.metadata,
    )


def save_metrics_json(metrics, output_path):
    """Save metrics as JSON without overwriting an existing file."""
    def jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return jsonable(value.tolist())
        if isinstance(value, np.generic):
            return jsonable(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metrics must contain finite JSON values")
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    payload = jsonable(metrics)
    output_path = Path(output_path)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(output_path, encoded, overwrite=False)
    return output_path


def _atomic_write_text(path, text, *, overwrite=False):
    """Write text through a same-directory temporary file and atomic replace."""
    path = Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return path


if __name__ == "__main__":
    main()
