"""Local AGG-only training implementation.

The module intentionally has no dependency on the parent project's trainer or
model namespace.  It preserves the original supervised three-source schedule
while using the isolated AGG_FWC model and datasets.
"""

import json
import hashlib
import logging
import math
import os
import platform
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from AGG_FWC.evaluate import compute_classification_metrics, save_metrics_json
from AGG_FWC.models.agg_fwc_model import AGGFWCModel


def collect_run_metadata(config):
    """Collect environment and seed metadata for one reproducible run."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "seed": config.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": str(torch.device("cuda" if cuda_available else "cpu")),
        "requested_cuda_device": config.cuda_device,
        "cuda_device_count": torch.cuda.device_count(),
        "require_cuda": config.require_cuda,
        "cuda_fallback": not cuda_available and not config.require_cuda,
    }


def validate_dataset_lengths(dataset_map):
    """Reject missing or empty source/target datasets before DataLoader setup."""
    for name in ("src_1", "src_2", "src_3", "tar"):
        if name not in dataset_map or len(dataset_map[name]) <= 0:
            raise ValueError(f"dataset {name!r} must exist and be non-empty")


class _SingleUseTargetLoader:
    """Explicit target boundary: the final target loader may be iterated once."""

    def __init__(self, loader):
        self._loader = loader
        self.iterations = 0

    def __iter__(self):
        if self.iterations:
            raise RuntimeError("target evaluation may consume the target loader only once")
        self.iterations += 1
        return iter(self._loader)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _component_hash(value):
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256_mapping(component_hashes):
    if not isinstance(component_hashes, dict) or not component_hashes:
        raise ValueError("component_hashes must be a non-empty mapping")
    validated = {}
    for name, value in component_hashes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("component hash names must be non-empty strings")
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("component hashes must be 64-character sha256 hex strings")
        validated[name] = value.lower()
    return validated


def _validate_component_files(component_hashes, component_files):
    if not isinstance(component_files, dict) or set(component_files) != set(component_hashes):
        raise ValueError("component_files must specify one existing file for every component hash")
    for name, expected in component_hashes.items():
        path = Path(component_files[name])
        if not path.is_file():
            raise ValueError(f"component file for {name!r} does not exist")
        actual = _file_hash(path)
        if actual != expected:
            raise RuntimeError(f"component hash mismatch for {name!r}")
    return {name: str(Path(path).resolve()) for name, path in component_files.items()}


def _audit_record_ids(value):
    """Extract record IDs from scalar IDs or protocol.py canonical records."""
    if isinstance(value, dict):
        if "record_id" in value:
            return [str(value["record_id"])]
        for key in ("record_ids", "records"):
            if key in value:
                return _audit_record_ids(value[key])
        raise RuntimeError("protocol audit split entries must expose record_id values")
    if isinstance(value, (list, tuple)):
        record_ids = []
        for entry in value:
            record_ids.extend(_audit_record_ids(entry))
        return record_ids
    return [str(value)]


def _canonical_audit_records(value):
    normalized = _jsonable(value)
    if isinstance(normalized, list):
        if all(isinstance(item, dict) and "record_id" in item for item in normalized):
            return sorted(normalized, key=lambda item: str(item["record_id"]))
        return sorted(normalized, key=str)
    return normalized


def _extract_audit_split_ids(protocol_audit):
    splits = protocol_audit["splits"]
    if not isinstance(splits, dict):
        raise RuntimeError("protocol audit splits must be a mapping")

    def select(candidates, role):
        for name in candidates:
            if name in splits:
                ids = _audit_record_ids(splits[name])
                if not ids or len(set(ids)) != len(ids):
                    raise RuntimeError(f"protocol audit {role} record_ids must be unique")
                return frozenset(ids)
        raise RuntimeError(f"protocol audit splits must include {role} records")

    extracted = {
        "train": select(("train", "source_train"), "train"),
        "val": select(("val", "source_val", "validation"), "val"),
        "target": select(("target", "test", "target_test"), "target"),
    }
    if extracted["train"] & extracted["val"]:
        raise RuntimeError("protocol audit train/val record_ids overlap")
    if (extracted["train"] | extracted["val"]) & extracted["target"]:
        raise RuntimeError("protocol audit source/target record_ids overlap")

    if "records" in protocol_audit:
        records = protocol_audit["records"]
        if isinstance(records, dict):
            role_keys = ("train", "source_train", "val", "source_val", "test", "target")
            if any(key in records for key in role_keys):
                record_ids = set()
                for key in role_keys:
                    if key in records:
                        record_ids.update(_audit_record_ids(records[key]))
            elif "record_id" in records or "record_ids" in records or "records" in records:
                record_ids = set(_audit_record_ids(records))
            else:
                record_ids = {str(key) for key in records}
        else:
            record_ids = set(_audit_record_ids(records))
        if record_ids != set().union(*extracted.values()):
            raise RuntimeError("protocol audit records do not match splits")
    return extracted


def _validate_protocol_audit(protocol_audit):
    required = {
        "status",
        "record_count",
        "fault_class_counts",
        "bearing_intersections",
        "file_hashes",
        "manifest_hash",
        "split_hashes",
        "splits",
    }
    if not isinstance(protocol_audit, dict) or protocol_audit.get("status") != "passed":
        raise RuntimeError("protocol audit status must be 'passed'")
    missing = required.difference(protocol_audit)
    if missing:
        raise RuntimeError("protocol audit missing " + ", ".join(sorted(missing)))
    if not isinstance(protocol_audit["file_hashes"], dict) or not protocol_audit["file_hashes"]:
        raise RuntimeError("protocol audit file_hashes must be a non-empty mapping")
    if not isinstance(protocol_audit["split_hashes"], dict) or not protocol_audit["split_hashes"]:
        raise RuntimeError("protocol audit split_hashes must be a non-empty mapping")
    hashes = [protocol_audit["manifest_hash"]]
    hashes.extend(protocol_audit["file_hashes"].values())
    hashes.extend(protocol_audit["split_hashes"].values())
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None for value in hashes):
        raise RuntimeError("protocol audit hashes must be 64-character sha256 hex strings")
    extracted = _extract_audit_split_ids(protocol_audit)
    split_values = protocol_audit["splits"]
    for name in ("train", "val", "test"):
        if name not in split_values:
            continue
        expected = _component_hash(_canonical_audit_records(split_values[name]))
        if expected != protocol_audit["split_hashes"].get(name, "").lower():
            raise RuntimeError(f"protocol audit split hash mismatch for {name!r}")
    all_records = []
    for name in sorted(split_values):
        all_records.extend(_canonical_audit_records(split_values[name]))
    if all(isinstance(item, dict) and "record_id" in item for item in all_records):
        all_records = sorted(all_records, key=lambda item: str(item["record_id"]))
    else:
        all_records = sorted(all_records, key=str)
    expected_manifest = _component_hash(all_records)
    if expected_manifest != protocol_audit["manifest_hash"].lower():
        raise RuntimeError("protocol audit manifest hash mismatch")
    for raw_path, expected in protocol_audit["file_hashes"].items():
        path = Path(raw_path)
        if not path.is_file() or _file_hash(path) != expected.lower():
            raise RuntimeError(f"protocol audit file hash mismatch for {raw_path!r}")
    try:
        json.dumps(_jsonable(protocol_audit), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("protocol audit must be finite JSON") from exc
    return extracted


def _validate_loader(loader, name, expected_partition):
    partition = getattr(loader, "partition", None)
    record_level = getattr(loader, "record_level", None)
    if partition != expected_partition or record_level is not True:
        raise ValueError(
            f"{name} must carry partition={expected_partition!r} and record_level=True"
        )
    record_ids = getattr(loader, "record_ids", None)
    if record_ids is None:
        raise ValueError(f"{name} must carry record_ids metadata")
    record_ids = np.asarray(record_ids)
    if record_ids.ndim != 1 or record_ids.size == 0:
        raise ValueError(f"{name} record_ids must be a non-empty one-dimensional sequence")
    normalized = [str(value) for value in record_ids.tolist()]
    if any(not value.strip() for value in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} record_ids must be unique and non-empty")
    return tuple(normalized)


def _atomic_write_text(path, text, *, overwrite=True):
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


def _atomic_savez(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".npz", dir=path.parent
        )
        os.close(descriptor)
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return path


def _atomic_copyfile(source, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return destination


class StrictStagedRunner:
    """Small, dependency-injected strict protocol runner for Task7.

    Source-stage callbacks receive only source train/record-validation loaders.
    The target loader is wrapped and is available only to the final evaluator.
    """

    STAGES = (
        "source_train_record_val",
        "calibrate_sources",
        "fit_attribute_predictor",
        "train_classifier_with_fwc_on_sources",
        "evaluate_target_once",
    )
    REQUIRED_DIAGNOSTIC_FIELDS = (
        "record_id",
        "initial_layout",
        "final_layout",
        "external_items",
        "red_priority",
        "selected_items",
        "collectibles",
        "eviction_reasons",
        "final_window_probabilities",
        "record_prediction",
        "inference_ms",
    )

    def __init__(
        self,
        config,
        source_train_loader,
        source_val_loader,
        target_loader,
        num_classes,
        source_split,
        target_split,
        protocol_audit,
        *,
        stage_callbacks=None,
        target_evaluator=None,
        component_hashes=None,
        component_files=None,
        protocol_mode=None,
    ):
        self.config = config
        self.protocol_mode = protocol_mode or getattr(config, "protocol_mode", "legacy")
        if self.protocol_mode != "strict" or getattr(config, "protocol_mode", None) != "strict":
            raise ValueError("StrictStagedRunner requires protocol_mode='strict'")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        audit_record_ids = _validate_protocol_audit(protocol_audit)
        self.component_hashes = _validate_sha256_mapping(component_hashes)
        self.component_files = _validate_component_files(self.component_hashes, component_files)
        self.source_train_record_ids = _validate_loader(
            source_train_loader, "source_train_loader", "source"
        )
        self.source_val_record_ids = _validate_loader(
            source_val_loader, "source_val_loader", "source"
        )
        self.target_record_ids = _validate_loader(target_loader, "target_loader", "target")
        train_ids = set(self.source_train_record_ids)
        val_ids = set(self.source_val_record_ids)
        target_ids = set(self.target_record_ids)
        if train_ids & val_ids:
            raise ValueError("source train/val record_ids must not overlap")
        if (train_ids | val_ids) & target_ids:
            raise ValueError("source and target record_ids must not overlap")
        if audit_record_ids["train"] != frozenset(train_ids):
            raise RuntimeError("protocol audit train records do not match source_train_loader")
        if audit_record_ids["val"] != frozenset(val_ids):
            raise RuntimeError("protocol audit val records do not match source_val_loader")
        if audit_record_ids["target"] != frozenset(target_ids):
            raise RuntimeError("protocol audit target records do not match target_loader")
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self._target_loader = _SingleUseTargetLoader(target_loader)
        self.num_classes = num_classes
        self.source_split = _jsonable(source_split)
        self.target_split = _jsonable(target_split)
        self.protocol_audit = _jsonable(protocol_audit)
        self.stage_callbacks = dict(stage_callbacks or {})
        self.target_evaluator = target_evaluator
        self.state = {"stage_results": {}, "stage_order": []}
        self.source_epoch_metrics = []
        self.protocol_audit_hash = _component_hash(self.protocol_audit)
        self.config_hash = None
        self.checkpoint_hash = None
        self._target_evaluation_started = False
        self._target_evaluation_finished = False
        self._failure_reason = None
        self._manifest_path = Path(self.config.result_dir) / "strict_manifest.json"

    @property
    def target_loader_iterations(self):
        return self._target_loader.iterations

    def _run_source_stage(self, stage_name):
        self._expect_next_stage(stage_name)
        callback = self.stage_callbacks.get(stage_name)
        if callback is None:
            raise RuntimeError(f"strict stage {stage_name} requires an injected callback")
        result = callback(self.source_train_loader, self.source_val_loader, self.state)
        if not isinstance(result, dict):
            raise ValueError(f"strict stage {stage_name} must return a mapping")
        self.state["stage_results"][stage_name] = result
        self.state["stage_order"].append(stage_name)
        try:
            self._write_manifest(status="source_stage_complete")
        except Exception as exc:
            self._mark_failed(exc)
            raise
        return result

    def _expect_next_stage(self, stage_name):
        completed = self.state["stage_order"]
        expected_index = len(completed)
        if expected_index >= len(self.STAGES) or self.STAGES[expected_index] != stage_name:
            expected = self.STAGES[expected_index] if expected_index < len(self.STAGES) else None
            raise RuntimeError(
                f"strict stage order violation: expected {expected!r}, got {stage_name!r}"
            )

    def source_train_record_val(self):
        self._expect_next_stage("source_train_record_val")
        callback = self.stage_callbacks.get("source_train_record_val")
        if callback is None:
            raise RuntimeError("strict source_train_record_val requires an injected callback")
        best = None
        for epoch in range(self.config.epoch):
            result = callback(
                self.source_train_loader,
                self.source_val_loader,
                epoch,
                self.state,
            )
            if not isinstance(result, dict):
                raise ValueError("source epoch callback must return a mapping")
            metric = result.get("val_metric")
            checkpoint = result.get("checkpoint_path")
            if not isinstance(metric, (int, float, np.number)) or not math.isfinite(float(metric)):
                raise ValueError("source validation metric must be finite")
            if checkpoint is None or not Path(checkpoint).is_file():
                raise ValueError("source epoch callback must return an existing checkpoint_path")
            if result.get("frozen") is not True:
                raise ValueError("source epoch checkpoint must be explicitly frozen")
            row = {
                "epoch": int(epoch),
                "val_metric": float(metric),
                "checkpoint_path": str(Path(checkpoint)),
            }
            self.source_epoch_metrics.append(row)
            if best is None or row["val_metric"] > best["val_metric"]:
                best = row
        if best is None:
            raise ValueError("source training must run at least one epoch")
        best_path = Path(self.config.checkpoint_dir) / "best.pt"
        chosen_path = Path(best["checkpoint_path"])
        if chosen_path.resolve() != best_path.resolve():
            try:
                _atomic_copyfile(chosen_path, best_path)
            except Exception as exc:
                self._mark_failed(exc)
                raise
        self.state["stage_results"]["source_train_record_val"] = {
            "checkpoint_path": best_path,
            "best_epoch": best["epoch"],
            "best_val_metric": best["val_metric"],
            "frozen": True,
        }
        self.state["stage_order"].append("source_train_record_val")
        try:
            self._write_manifest(status="source_stage_complete")
        except Exception as exc:
            self._mark_failed(exc)
            raise
        return self.state["stage_results"]["source_train_record_val"]

    def train_agg_on_sources(self):
        """Compatibility alias; strict manifests use source_train_record_val."""
        return self.source_train_record_val()

    def calibrate_sources(self):
        return self._run_source_stage("calibrate_sources")

    def fit_attribute_predictor(self):
        return self._run_source_stage("fit_attribute_predictor")

    def train_classifier_with_fwc_on_sources(self):
        return self._run_source_stage("train_classifier_with_fwc_on_sources")

    def run_source_stages(self):
        for stage_name in self.STAGES[:-1]:
            getattr(self, stage_name)()
        return self.state

    def _write_manifest(self, *, status, output_path=None, target_evaluation_count=None):
        config_path = Path(self.config.result_dir) / "config.json"
        payload = {
            "status": status,
            "protocol_mode": self.protocol_mode,
            "fwc_stage": self.config.fwc_stage,
            "seed": self.config.seed,
            "config_path": str(config_path),
            "source_split": self.source_split,
            "target_split": self.target_split,
            "protocol_audit": self.protocol_audit,
            "component_hashes": self.component_hashes,
            "component_files": self.component_files,
            "source_train_record_ids": list(self.source_train_record_ids),
            "source_val_record_ids": list(self.source_val_record_ids),
            "target_record_ids": list(self.target_record_ids),
            "stages": list(self.state["stage_order"]),
            "source_epoch_metrics": self.source_epoch_metrics,
            "best_epoch": self.state["stage_results"].get("source_train_record_val", {}).get("best_epoch"),
            "best_val_metric": self.state["stage_results"].get("source_train_record_val", {}).get("best_val_metric"),
            "config_hash": self.config_hash,
            "protocol_audit_hash": self.protocol_audit_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "target_loader_iterations": self.target_loader_iterations,
            "target_evaluation_count": int(
                self._target_evaluation_finished
                if target_evaluation_count is None
                else target_evaluation_count
            ),
        }
        if self._failure_reason is not None:
            payload["failure_reason"] = self._failure_reason
        encoded = json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False
        )
        destination = self._manifest_path if output_path is None else Path(output_path)
        _atomic_write_text(destination, encoded, overwrite=True)
        return destination

    def _mark_failed(self, exc):
        self._failure_reason = f"{type(exc).__name__}: {exc}"
        try:
            self._write_manifest(status="failed")
        except Exception:
            logging.exception("could not write strict failed manifest")

    def _source_artifact(self, key):
        for stage_name in self.STAGES:
            result = self.state["stage_results"].get(stage_name, {})
            if key in result:
                return result[key]
        return None

    def _require_finalization(self):
        checkpoint = self._source_artifact("checkpoint_path")
        checkpoint_result = self.state["stage_results"].get("source_train_record_val", {})
        if (
            checkpoint is None
            or not Path(checkpoint).is_file()
            or checkpoint_result.get("frozen") is not True
        ):
            raise RuntimeError("frozen source checkpoint is required before target evaluation")
        if self.state["stage_order"] != list(self.STAGES[:-1]):
            raise RuntimeError("strict source stages must complete before target evaluation")
        attribute_result = self.state["stage_results"].get("fit_attribute_predictor", {})
        if (
            self._source_artifact("attribute_predictor") is None
            or attribute_result.get("frozen") is not True
        ):
            raise RuntimeError("frozen source attribute predictor is required before target evaluation")
        classifier_result = self.state["stage_results"].get(
            "train_classifier_with_fwc_on_sources", {}
        )
        if (
            self._source_artifact("classifier") is None
            or classifier_result.get("frozen") is not True
        ):
            raise RuntimeError("frozen source classifier is required before target evaluation")
        calibration_result = self.state["stage_results"].get("calibrate_sources", {})
        if calibration_result.get("frozen") is not True:
            raise RuntimeError("frozen source calibration is required before target evaluation")
        _validate_protocol_audit(self.protocol_audit)
        _validate_component_files(self.component_hashes, self.component_files)
        config_path = Path(self.config.result_dir) / "config.json"
        if not config_path.is_file():
            raise RuntimeError("complete config snapshot is required before target evaluation")
        return Path(checkpoint)

    def _validate_target_result(self, result):
        if not isinstance(result, dict):
            raise ValueError("target evaluator must return a mapping")
        required = {"record_ids", "labels", "probabilities", "predictions"}
        missing = required.difference(result)
        if missing:
            raise ValueError("target evaluator is missing " + ", ".join(sorted(missing)))
        record_ids = np.asarray(result["record_ids"])
        labels = np.asarray(result["labels"])
        probabilities = np.asarray(result["probabilities"], dtype=float)
        predictions = np.asarray(result["predictions"])
        if record_ids.ndim != 1 or labels.ndim != 1 or predictions.ndim != 1:
            raise ValueError("target record outputs must be one-dimensional")
        if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(
            predictions.dtype, np.integer
        ):
            raise ValueError("target labels and predictions must be integer labels")
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError("target probabilities must have one column per class")
        if not (len(record_ids) == len(labels) == len(predictions) == len(probabilities)):
            raise ValueError("target record outputs must have the same length")
        normalized_ids = [str(value) for value in record_ids.tolist()]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("target record_ids must be unique")
        if set(normalized_ids) != set(self.target_record_ids):
            raise ValueError("target record outputs must cover every target record exactly once")
        if not np.isfinite(probabilities).all():
            raise ValueError("target probabilities must be finite")
        return (
            np.asarray(normalized_ids, dtype=str),
            labels,
            probabilities,
            predictions,
        )

    def _save_fwc_diagnostics(self, result, record_ids, elapsed_ms, *, output_path=None):
        diagnostics = result.get("fwc_diagnostics", result.get("diagnostics", []))
        if isinstance(diagnostics, dict):
            diagnostics = [diagnostics]
        diagnostics = list(diagnostics or [])
        if len(diagnostics) != len(record_ids):
            raise ValueError("fwc diagnostics must contain one row per target record")
        path = (
            Path(self.config.result_dir) / "fwc_diagnostics.jsonl"
            if output_path is None
            else Path(output_path)
        )
        encoded_rows = []
        for index, item in enumerate(diagnostics):
            if not isinstance(item, dict):
                raise ValueError("fwc diagnostics rows must be JSON objects")
            missing = set(self.REQUIRED_DIAGNOSTIC_FIELDS).difference(item)
            if missing:
                raise ValueError("fwc diagnostics missing " + ", ".join(sorted(missing)))
            selected_items = item["selected_items"]
            if not isinstance(selected_items, list) or any(
                not isinstance(selected, dict)
                or not {"raw_utility", "effective_utility", "multiplier"}.issubset(selected)
                for selected in selected_items
            ):
                raise ValueError(
                    "fwc diagnostics selected_items must include raw/effective utility and multiplier"
                )
            for selected in selected_items:
                for key in ("raw_utility", "effective_utility", "multiplier"):
                    if not isinstance(selected[key], (int, float)) or not math.isfinite(
                        float(selected[key])
                    ):
                        raise ValueError("fwc diagnostics utilities must be finite")
            if (
                not isinstance(item["record_prediction"], (int, np.integer))
                or isinstance(item["record_prediction"], (bool, np.bool_))
            ):
                raise ValueError("fwc diagnostic record_prediction must be an integer label")
            if item["record_id"] != str(record_ids[index]):
                raise ValueError("fwc diagnostic record_id does not match record predictions")
            inference_ms = item["inference_ms"]
            if not isinstance(inference_ms, (int, float)) or not math.isfinite(float(inference_ms)):
                raise ValueError("fwc diagnostics inference_ms must be finite")
            try:
                encoded_rows.append(json.dumps(item, sort_keys=True, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise ValueError("fwc diagnostics must be finite JSON") from exc
        _atomic_write_text(path, "\n".join(encoded_rows) + "\n", overwrite=False)
        return path

    def _publish_target_artifacts(self, temporary_dir):
        """Publish the staged target bundle with rollback on any replacement error."""
        temporary_dir = Path(temporary_dir)
        names = (
            "record_predictions.npz",
            "fwc_diagnostics.jsonl",
            "metrics.json",
            "strict_manifest.json",
        )
        backup_dir = temporary_dir / ".backup"
        backup_dir.mkdir()
        backups = {}
        published = []
        try:
            for name in names:
                staged = temporary_dir / name
                destination = Path(self.config.result_dir) / name
                if not staged.is_file():
                    raise RuntimeError(f"staged target artifact is missing: {name}")
                if destination.exists():
                    backup = backup_dir / name
                    os.replace(destination, backup)
                    backups[name] = backup
                os.replace(staged, destination)
                published.append(name)
        except Exception:
            for name in reversed(published):
                destination = Path(self.config.result_dir) / name
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            for name, backup in backups.items():
                destination = Path(self.config.result_dir) / name
                if backup.exists():
                    try:
                        os.replace(backup, destination)
                    except Exception:
                        logging.exception("could not restore strict artifact %s", name)
            raise

    def evaluate_target_once(self):
        if self._target_evaluation_started:
            raise RuntimeError("target evaluation may be called only once")
        checkpoint = self._require_finalization()
        config_path = Path(self.config.result_dir) / "config.json"
        if self.target_evaluator is None:
            raise RuntimeError("strict runner requires an explicit target evaluator")
        self.config_hash = _file_hash(config_path)
        self.checkpoint_hash = _file_hash(checkpoint)
        self._target_evaluation_started = True
        temporary_dir = None
        stage_added = False
        try:
            self._write_manifest(status="target_evaluation_started")
            started = time.perf_counter()
            result = self.target_evaluator(self._target_loader, self.state)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self.target_loader_iterations != 1:
                raise RuntimeError("target evaluator must iterate the target loader exactly once")
            record_ids, labels, probabilities, predictions = self._validate_target_result(result)
            metrics = compute_classification_metrics(labels, predictions, self.num_classes)
            metrics.update(
                {
                    "run_id": self.config.run_id,
                    "config_path": config_path,
                    "checkpoint_path": checkpoint,
                    "protocol_mode": self.protocol_mode,
                    "target_evaluation_count": 1,
                    "metadata": collect_run_metadata(self.config),
                }
            )
            temporary_dir = Path(
                tempfile.mkdtemp(prefix=".strict-final-", dir=self.config.result_dir)
            )
            _atomic_savez(
                temporary_dir / "record_predictions.npz",
                record_ids=record_ids,
                labels=labels,
                probabilities=probabilities,
                predictions=predictions,
            )
            self._save_fwc_diagnostics(
                result,
                record_ids,
                elapsed_ms,
                output_path=temporary_dir / "fwc_diagnostics.jsonl",
            )
            save_metrics_json(metrics, temporary_dir / "metrics.json")
            self._target_evaluation_finished = True
            self.state["stage_order"].append("evaluate_target_once")
            stage_added = True
            self._write_manifest(
                status="complete",
                output_path=temporary_dir / "strict_manifest.json",
                target_evaluation_count=1,
            )
            self._publish_target_artifacts(temporary_dir)
            return {"metrics": metrics, "record_predictions": record_ids}
        except Exception as exc:
            if stage_added and self.state["stage_order"][-1] == "evaluate_target_once":
                self.state["stage_order"].pop()
            self._target_evaluation_finished = False
            self._mark_failed(exc)
            raise
        finally:
            if temporary_dir is not None:
                shutil.rmtree(temporary_dir, ignore_errors=True)

    def run(self):
        from AGG_FWC.config import save_config_snapshot

        config_path = Path(self.config.result_dir) / "config.json"
        if not config_path.exists():
            save_config_snapshot(self.config)
        self.config_hash = _file_hash(config_path)
        try:
            self._write_manifest(status="created")
            self.run_source_stages()
            return self.evaluate_target_once()
        except Exception as exc:
            self._mark_failed(exc)
            raise


StagedRunner = StrictStagedRunner


class AGGTrainer:
    def __init__(self, config):
        self.config = config
        if getattr(config, "protocol_mode", "legacy") == "strict":
            raise RuntimeError(
                "strict protocol must use StrictStagedRunner; AGGTrainer is legacy-only"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_count = torch.cuda.device_count() if self.device.type == "cuda" else 1
        self._configure_logging()
        try:
            self.metadata = collect_run_metadata(config)
            for key, value in self.metadata.items():
                logging.info("%s=%s", key, value)
            if config.require_cuda and not self.metadata["cuda_available"]:
                raise RuntimeError(
                    "CUDA is required by this configuration but is unavailable"
                )
            self.datasets = self._load_datasets()
            validate_dataset_lengths(self.datasets)
            self.dataloaders = {
                name: torch.utils.data.DataLoader(
                    self.datasets[name],
                    batch_size=config.batch_size,
                    shuffle=name.startswith("src"),
                    num_workers=config.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                for name in ("src_1", "src_2", "src_3", "tar")
            }
            self.model = AGGFWCModel(num_classes=self.datasets["num_cls"]).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
            steps = [int(step.strip()) for step in config.steps.split(",")]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, steps, gamma=config.gamma
            )
        except Exception:
            logging.exception("AGGTrainer initialization failed")
            self._close_log_handler()
            raise

    def _configure_logging(self):
        self.log_path = Path(self.config.log_dir) / "train.log"
        logger = logging.getLogger()
        for existing in list(logger.handlers):
            if isinstance(existing, logging.FileHandler) and Path(existing.baseFilename) == self.log_path:
                logger.removeHandler(existing)
                existing.close()
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        self._log_handler = handler

    def _close_log_handler(self):
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            logger = logging.getLogger()
            logger.removeHandler(handler)
            handler.close()
            self._log_handler = None

    def _load_datasets(self):
        from AGG_FWC.datasets import PHM2009_DG, PU_DG

        dataset_class = PHM2009_DG if self.config.data_name == "PHM2009" else PU_DG
        dataset = dataset_class(
            transfer_task=self.config.transfer_task,
            normalizetype=self.config.normalizetype,
            num_train_samples=self.config.num_train_samples,
            num_test_samples=self.config.num_test_samples,
        )
        dataset.data_dir = str(self.config.data_root)
        src_1, src_2, src_3, target, num_cls = dataset.train_test_data_split()
        return {"src_1": src_1, "src_2": src_2, "src_3": src_3, "tar": target, "num_cls": num_cls}

    def train(self):
        epoch_metrics = []
        try:
            for epoch in range(self.config.epoch):
                self.model.train()
                train_loss, train_count = 0.0, 0
                iter_src_2 = iter(self.dataloaders["src_2"])
                iter_src_3 = iter(self.dataloaders["src_3"])
                for src_1_inputs, src_1_labels in self.dataloaders["src_1"]:
                    src_2_inputs, src_2_labels = next(iter_src_2)
                    src_3_inputs, src_3_labels = next(iter_src_3)
                    inputs = torch.cat((src_1_inputs, src_2_inputs, src_3_inputs), dim=0).to(self.device)
                    labels = torch.cat((src_1_labels, src_2_labels, src_3_labels), dim=0).to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.criterion(self.model(inputs), labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * labels.size(0)
                    train_count += labels.size(0)

                self.model.eval()
                test_loss, correct, test_count = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in self.dataloaders["tar"]:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        test_loss += loss.item() * labels.size(0)
                        correct += (logits.argmax(dim=1) == labels).float().sum().item()
                        test_count += labels.size(0)
                self.lr_scheduler.step()
                row = {
                    "epoch": epoch,
                    "train_loss": float(np.divide(train_loss, train_count)),
                    "test_loss": float(np.divide(test_loss, test_count)),
                    "test_accuracy": float(np.divide(correct, test_count)),
                }
                epoch_metrics.append(row)
                logging.info("epoch=%s metrics=%s", epoch, row)

            final_labels = []
            final_predictions = []
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.dataloaders["tar"]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(inputs)
                    final_labels.append(labels)
                    final_predictions.append(logits.argmax(dim=1))

            labels_cpu = torch.cat(final_labels, dim=0).detach().cpu().numpy()
            predictions_cpu = torch.cat(final_predictions, dim=0).detach().cpu().numpy()
            classification_metrics = compute_classification_metrics(
                labels_cpu, predictions_cpu, num_classes=self.datasets["num_cls"]
            )
            from AGG_FWC.config import save_config_snapshot

            save_config_snapshot(self.config)
            torch.save(self.model.state_dict(), Path(self.config.checkpoint_dir) / "final.pt")
            classification_metrics.update(
                {
                    "run_id": self.config.run_id,
                    "config_path": Path(self.config.result_dir) / "config.json",
                    "checkpoint_path": Path(self.config.checkpoint_dir) / "final.pt",
                }
            )
            np.savez(
                Path(self.config.result_dir) / "predictions.npz",
                labels=labels_cpu,
                predictions=predictions_cpu,
            )
            epoch_metrics_path = Path(self.config.result_dir) / "epoch_metrics.json"
            epoch_metrics_path.write_text(json.dumps(epoch_metrics, indent=2), encoding="utf-8")
            save_metrics_json(
                classification_metrics, Path(self.config.result_dir) / "metrics.json"
            )
            return epoch_metrics
        except Exception:
            logging.exception("AGG training failed")
            raise
        finally:
            self._close_log_handler()


Train_utils = AGGTrainer
