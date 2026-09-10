"""Real source-only staged runtime for the PU FWC experiment.

This adapter intentionally keeps the existing ``StrictStagedRunner`` small.
It owns the concrete PU record reader, the five stage callbacks, and the
record-level target evaluator.  The target loader is never iterated by any
source-stage callback.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

from AGG_FWC.datasets.strict_pu_adapter import build_strict_pu_loaders
from AGG_FWC.evaluate import compute_classification_metrics
from AGG_FWC.models.agg_fwc_model import AGGFWCModel
from AGG_FWC.models.classifier import Classifier
from AGG_FWC.models.feature_combination import FWCFeatureCombination
from AGG_FWC.models.fwc_attributes import AttributePredictor, fit_attribute_predictor
from AGG_FWC.models.fwc_calibration import (
    calibrate_condition,
    create_audited_source_payload,
    save_condition_attribute_report,
    select_external_features,
    single_feature_grouped_oof_accuracies,
)
from AGG_FWC.models.fwc_types import ConditionAttributeReport
from AGG_FWC.protocol import create_audited_source_partition
from AGG_FWC.strict_pu_experiment import build_audited_window_loader


STAGES = (
    "source_train_record_val",
    "calibrate_sources",
    "fit_attribute_predictor",
    "train_classifier_with_fwc_on_sources",
    "evaluate_target_once",
)
NUM_CLASSES = 14


def required_stage_names():
    """Return the exact staged-runner contract."""

    return STAGES


def calibrate_condition_from_model(
    classifier,
    features,
    labels,
    record_ids,
    *,
    condition_id,
    seed,
    device,
    repeats=5,
    mask_batch_size=32,
):
    """Calibrate C/S/R by batched ablation of the frozen AGG classifier.

    The definition remains the agreed one: contribution is the macro-F1 drop
    after masking one feature, stability is the inverse spread over five
    grouped resamples, and redundancy is correlation with higher-contribution
    features.  Using the already frozen classifier avoids fitting thousands of
    auxiliary models and keeps the ablation aligned with the final predictor.
    """

    values = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    record_ids = np.asarray(record_ids)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("features must be a finite non-empty matrix")
    if labels.ndim != 1 or record_ids.ndim != 1 or len(labels) != len(values) or len(record_ids) != len(values):
        raise ValueError("features, labels, and record IDs must have matching lengths")
    if np.unique(labels).size < 2 or np.unique(record_ids).size < 2:
        raise ValueError("model calibration requires at least two classes and two records")
    if repeats != 5:
        raise ValueError("repeats must be exactly five")
    if mask_batch_size <= 0:
        raise ValueError("mask_batch_size must be positive")

    classifier = classifier.to(device)
    classifier.eval()
    contribution, delta_margins = _model_contribution(
        classifier, values, labels, device=device, mask_batch_size=mask_batch_size
    )
    normalized_contribution = _unit_interval(contribution)
    resampled = []
    groups = np.unique(record_ids)
    for repeat in range(5):
        rng = np.random.default_rng(int(seed) + repeat)
        count = max(2, int(np.ceil(groups.size * 0.8)))
        selected = rng.choice(groups, size=min(count, groups.size), replace=False)
        indices = np.flatnonzero(np.isin(record_ids, selected))
        if np.unique(labels[indices]).size < np.unique(labels).size:
            indices = np.arange(values.shape[0])
        resampled.append(
            _model_contribution(
                classifier,
                values[indices],
                labels[indices],
                device=device,
                mask_batch_size=mask_batch_size,
            )[0]
        )
    spread = np.std(np.stack(resampled), axis=0)
    lower, upper = float(spread.min()), float(spread.max())
    stability = np.ones_like(spread) if np.isclose(lower, upper) else 1.0 - (spread - lower) / (upper - lower)
    redundancy = np.zeros(values.shape[1], dtype=float)
    for index, score in enumerate(normalized_contribution):
        higher = np.flatnonzero(normalized_contribution > score)
        if higher.size:
            redundancy[index] = max(
                abs(_pearson(values[:, index], values[:, other])) for other in higher
            )
    return ConditionAttributeReport(
        condition_id=str(condition_id),
        contribution=normalized_contribution,
        stability=np.clip(stability, 0.0, 1.0),
        redundancy=np.clip(redundancy, 0.0, 1.0),
        c_quantiles=tuple(np.quantile(normalized_contribution, [0.05, 0.95]).tolist()),
        delta_margin_quantiles=tuple(np.quantile(delta_margins, [0.05, 0.95]).tolist()),
    )


def _model_contribution(classifier, features, labels, *, device, mask_batch_size):
    feature_count = features.shape[1]
    with torch.no_grad():
        base = torch.as_tensor(features, dtype=torch.float32, device=device)
        baseline_probabilities = torch.softmax(classifier(base), dim=1).cpu().numpy()
    baseline_predictions = baseline_probabilities.argmax(axis=1)
    classes = np.unique(labels)
    from sklearn.metrics import f1_score

    baseline_score = f1_score(
        labels, baseline_predictions, labels=classes, average="macro", zero_division=0
    )
    contribution = np.empty(feature_count, dtype=float)
    all_deltas = []
    for start in range(0, feature_count, mask_batch_size):
        stop = min(start + mask_batch_size, feature_count)
        count = stop - start
        masked = base.unsqueeze(0).repeat(count, 1, 1)
        local = torch.arange(count, device=device)
        masked[local, :, torch.arange(start, stop, device=device)] = 0.0
        with torch.no_grad():
            probabilities = torch.softmax(classifier(masked.reshape(-1, feature_count)), dim=1)
        probabilities = probabilities.reshape(count, features.shape[0], -1).cpu().numpy()
        for local_index in range(count):
            masked_predictions = probabilities[local_index].argmax(axis=1)
            contribution[start + local_index] = baseline_score - f1_score(
                labels,
                masked_predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )
            confidence = probabilities[local_index, np.arange(labels.size), baseline_predictions]
            all_deltas.extend(np.maximum(0.0, baseline_probabilities.max(axis=1) - confidence))
    return contribution, np.asarray(all_deltas, dtype=float)


def _unit_interval(values):
    values = np.asarray(values, dtype=float)
    lower, upper = values.min(), values.max()
    return np.zeros_like(values) if np.isclose(lower, upper) else (values - lower) / (upper - lower)


def _pearson(first, second):
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return 0.0 if not np.isfinite(value) else value


def fast_single_feature_grouped_oof_accuracies(features, labels, record_ids, seed=2026):
    """Compute the approved one-feature OOF rule with bounded parallelism."""

    values = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    record_ids = np.asarray(record_ids)
    if values.ndim != 2 or values.shape[0] != labels.size or labels.size != record_ids.size:
        raise ValueError("features, labels, and record IDs must have matching lengths")
    groups = np.unique(record_ids)
    if groups.size < 2 or np.unique(labels).size < 2:
        raise ValueError("external feature screening requires two groups and two classes")
    splits = tuple(GroupKFold(n_splits=min(5, groups.size)).split(values, labels, record_ids))

    def score_one(feature_index):
        predictions = np.empty(labels.size, dtype=np.int64)
        for train_indices, test_indices in splits:
            classifier = LogisticRegression(
                C=1.0,
                max_iter=200,
                solver="lbfgs",
                random_state=int(seed) + int(feature_index),
            )
            classifier.fit(values[train_indices][:, [feature_index]], labels[train_indices])
            predictions[test_indices] = classifier.predict(
                values[test_indices][:, [feature_index]]
            )
        return float(accuracy_score(labels, predictions))

    workers = min(8, max(1, len(values) // 64))
    scores = Parallel(n_jobs=workers, prefer="threads")(
        delayed(score_one)(index) for index in range(values.shape[1])
    )
    return np.asarray(scores, dtype=float)


def build_strict_pu_fwc_dependencies(config, manifest_path, audit_path):
    """Build concrete dependencies for ``StrictStagedRunner`` on PU T2.

    The generated audit is explicitly the paper's condition-transfer protocol:
    bearings may reappear across conditions, so this is not claimed as unseen-
    bearing domain generalization.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    audit_file = Path(audit_path).expanduser().resolve()
    manifest = pd.read_csv(manifest_file)
    bundle = build_strict_pu_loaders(
        data_root=config.data_root,
        manifest_path=manifest_file,
        audit_path=audit_file,
        protocol_mode="paper_condition",
        bearing_disjoint=False,
    )
    source_frame = manifest.loc[manifest["split"] == "source_train"].copy()
    val_frame = manifest.loc[manifest["split"] == "source_val"].copy()
    target_frame = manifest.loc[manifest["split"] == "target"].copy()
    loaders = {
        "source_train_loader": build_audited_window_loader(
            source_frame,
            partition="source",
            batch_size=config.batch_size,
            windows_per_record=8,
            normalizetype=config.normalizetype,
            num_workers=config.num_workers,
        ),
        "source_val_loader": build_audited_window_loader(
            val_frame,
            partition="source",
            batch_size=config.batch_size,
            windows_per_record=8,
            normalizetype=config.normalizetype,
            num_workers=config.num_workers,
        ),
        "target_loader": build_audited_window_loader(
            target_frame,
            partition="target",
            batch_size=config.batch_size,
            windows_per_record=8,
            normalizetype=config.normalizetype,
            num_workers=config.num_workers,
        ),
    }
    runtime = StrictPUFWCRuntime(
        config=config,
        manifest=manifest,
        protocol_audit=json.loads(audit_file.read_text(encoding="utf-8")),
        loaders=loaders,
        source_conditions=tuple(sorted(source_frame["condition_id"].unique())),
        target_condition=str(target_frame["condition_id"].iloc[0]),
        manifest_path=manifest_file,
        audit_path=audit_file,
    )
    return runtime.dependencies(bundle)


class StrictPUFWCRuntime:
    """Concrete implementation of the source-only FWC staged protocol."""

    def __init__(
        self,
        *,
        config,
        manifest: pd.DataFrame,
        protocol_audit: dict[str, Any],
        loaders: dict[str, Any],
        source_conditions: tuple[str, ...],
        target_condition: str,
        manifest_path: Path,
        audit_path: Path,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.protocol_audit = protocol_audit
        self.loaders = loaders
        self.source_conditions = source_conditions
        self.target_condition = target_condition
        self.manifest_path = manifest_path
        self.audit_path = audit_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.require_cuda and self.device.type != "cuda":
            raise RuntimeError("CUDA is required for the FWC run")
        self.model = AGGFWCModel(num_classes=NUM_CLASSES).to(self.device)
        self.fwc = None
        self.reports = {}
        self.external_indices = frozenset()
        self.grouped_features: dict[str, dict[str, Any]] = {}
        self.source_record_table: dict[str, Any] | None = None
        self.pooled_report: dict[str, tuple[float, float]] | None = None
        self._source_gate_cache: dict[str, torch.Tensor] = {}

    def dependencies(self, bundle):
        component_paths = {
            "manifest": self.manifest_path,
            "protocol_audit": self.audit_path,
            "runtime": Path(__file__),
            "feature_combination": Path(__file__).with_name("models") / "feature_combination.py",
            "extractor": Path(__file__).with_name("models") / "extractor.py",
            "classifier": Path(__file__).with_name("models") / "classifier.py",
        }
        component_files = {name: str(path.resolve()) for name, path in component_paths.items()}
        component_hashes = {name: _sha256(path) for name, path in component_paths.items()}
        return {
            "source_train_loader": self.loaders["source_train_loader"],
            "source_val_loader": self.loaders["source_val_loader"],
            "target_loader": self.loaders["target_loader"],
            "num_classes": NUM_CLASSES,
            "source_split": {
                "record_ids": list(self.loaders["source_train_loader"].record_ids),
                "validation_record_ids": list(self.loaders["source_val_loader"].record_ids),
                "protocol_mode": self.protocol_audit.get("protocol_mode"),
            },
            "target_split": {
                "record_ids": list(self.loaders["target_loader"].record_ids),
                "condition_id": self.target_condition,
            },
            "protocol_audit": self.protocol_audit,
            "component_hashes": component_hashes,
            "component_files": component_files,
            "stage_callbacks": {
                "source_train_record_val": self.source_train_record_val,
                "calibrate_sources": self.calibrate_sources,
                "fit_attribute_predictor": self.fit_attribute_predictor,
                "train_classifier_with_fwc_on_sources": self.train_classifier_with_fwc,
            },
            "target_evaluator": self.evaluate_target,
        }

    def source_train_record_val(self, train_loader, val_loader, epoch, state):
        """Train the original AGG model and select by source records only."""

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        if not hasattr(self, "source_optimizer"):
            self.source_optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        train_loss = 0.0
        train_count = 0
        for inputs, labels, _record_ids, _conditions in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = torch.as_tensor(labels, device=self.device, dtype=torch.long)
            self.source_optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            self.source_optimizer.step()
            train_loss += float(loss.detach()) * labels.numel()
            train_count += labels.numel()
        val_metrics = self._evaluate_loader_identity(val_loader)
        checkpoint = Path(self.config.checkpoint_dir) / f"source_epoch_{int(epoch):03d}.pt"
        torch.save(
            {"model_state": self.model.state_dict(), "epoch": int(epoch)},
            checkpoint,
        )
        return {
            "checkpoint_path": checkpoint,
            "val_metric": float(val_metrics["macro_f1"]),
            "train_loss": train_loss / max(train_count, 1),
            "val_metrics": val_metrics,
            "frozen": True,
        }

    def calibrate_sources(self, train_loader, val_loader, state):
        """Fit C/S/R and external-feature rules from source train records only."""

        self._load_best_source_model()
        grouped = self._extract_grouped_features(train_loader, self.model)
        self.grouped_features = grouped
        table = self._mean_record_table(grouped)
        self.source_record_table = table
        report_paths = {}
        accuracies = []
        for condition_index, condition_id in enumerate(self.source_conditions):
            selected = np.asarray(table["condition_ids"]) == condition_id
            report = calibrate_condition_from_model(
                self.model.classifier,
                table["features"][selected],
                table["labels"][selected],
                table["record_ids"][selected],
                condition_id=condition_id,
                seed=int(self.config.seed) + condition_index,
                device=self.device,
            )
            self.reports[condition_id] = report
            output = Path(self.config.result_dir) / f"fwc_calibration_{condition_index}.npz"
            save_condition_attribute_report(report, output)
            report_paths[condition_id] = output
            accuracies.append(
                fast_single_feature_grouped_oof_accuracies(
                    table["features"][selected],
                    table["labels"][selected],
                    table["record_ids"][selected],
                    seed=int(self.config.seed) + condition_index,
                )
            )
        condition_accuracies = np.stack(accuracies, axis=0)
        self.external_indices = frozenset(
            np.flatnonzero(select_external_features(condition_accuracies)).tolist()
        )
        np.save(
            Path(self.config.result_dir) / "external_feature_indices.npy",
            np.asarray(sorted(self.external_indices), dtype=np.int64),
        )
        self.pooled_report = {
            "c_quantiles": tuple(
                np.median([report.c_quantiles for report in self.reports.values()], axis=0)
            ),
            "delta_margin_quantiles": tuple(
                np.median(
                    [report.delta_margin_quantiles for report in self.reports.values()], axis=0
                )
            ),
        }
        return {
            "reports": self.reports,
            "report_paths": report_paths,
            "external_indices": sorted(self.external_indices),
            "condition_accuracies": condition_accuracies,
            "frozen": True,
        }

    def fit_attribute_predictor(self, train_loader, val_loader, state):
        """Fit and freeze H(z) using the already audited source table."""

        if self.source_record_table is None or not self.reports:
            raise RuntimeError("source calibration must precede attribute prediction")
        table = self.source_record_table
        source_features = torch.as_tensor(
            table["features"], dtype=torch.float32, device=self.device
        )
        partition = create_audited_source_partition(
            self.manifest,
            self.source_conditions,
            (self.target_condition,),
        )
        reports = tuple(self.reports[condition] for condition in self.source_conditions)
        payload = create_audited_source_payload(
            partition,
            source_features,
            table["condition_ids"],
            table["record_ids"],
            reports,
        )
        predictor = AttributePredictor(
            feature_dim=512, hidden_dim=int(self.config.attribute_hidden_dim)
        ).to(self.device)
        held_out = str(table["record_ids"][-1])
        fit_attribute_predictor(
            predictor=predictor,
            source_features=source_features,
            source_condition_ids=table["condition_ids"],
            attribute_reports=reports,
            source_record_ids=table["record_ids"],
            held_out_record_id=held_out,
            audited_source_payload=payload,
            epochs=int(self.config.attribute_epoch),
            lr=float(self.config.attribute_lr),
            seed=int(self.config.seed),
        )
        self.fwc = FWCFeatureCombination(
            predictor,
            external_indices=self.external_indices,
            source_reports=self.reports,
        )
        predictor_path = Path(self.config.checkpoint_dir) / "attribute_predictor.pt"
        torch.save({"model_state": predictor.state_dict()}, predictor_path)
        return {
            "attribute_predictor": predictor,
            "attribute_predictor_path": predictor_path,
            "frozen": True,
        }

    def train_classifier_with_fwc(self, train_loader, val_loader, state):
        """Freeze the extractor and train only the FWC-gated classifier."""

        if self.fwc is None or not self.grouped_features:
            raise RuntimeError("FWC attribute predictor must precede classifier training")
        self._load_best_source_model()
        self.model.feature_combination = self.fwc
        for parameter in self.model.extractor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.feature_combination.parameters():
            parameter.requires_grad_(False)
        optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        source_groups = self.grouped_features
        validation_groups = self._extract_grouped_features(val_loader, self.model)
        best = None
        for epoch in range(int(self.config.epoch)):
            self.model.classifier.train()
            total_loss = 0.0
            total_count = 0
            for record_id in sorted(source_groups):
                entry = source_groups[record_id]
                features = torch.as_tensor(entry["features"], dtype=torch.float32, device=self.device)
                labels = torch.full(
                    (features.shape[0],), int(entry["label"]), dtype=torch.long, device=self.device
                )
                gate = self._cached_source_gate(features, record_id)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model.classifier(features * gate)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * labels.numel()
                total_count += labels.numel()
            val_metrics = self._evaluate_grouped(validation_groups, use_collectibles=False)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
                "val_metric": float(val_metrics["macro_f1"]),
                "val_metrics": val_metrics,
            }
            if best is None or row["val_metric"] > best["val_metric"]:
                best = row
                checkpoint = Path(self.config.checkpoint_dir) / "fwc_classifier.pt"
                torch.save(
                    {"classifier_state": self.model.classifier.state_dict(), "epoch": epoch},
                    checkpoint,
                )
        if best is None:
            raise RuntimeError("FWC classifier did not complete an epoch")
        return {
            "classifier": Path(self.config.checkpoint_dir) / "fwc_classifier.pt",
            "best_epoch": best["epoch"],
            "best_val_metric": best["val_metric"],
            "frozen": True,
        }

    def evaluate_target(self, target_loader, state):
        """Consume target exactly once, then emit one result per target record."""

        if self.fwc is None or self.pooled_report is None:
            raise RuntimeError("source-only FWC stages must complete before target evaluation")
        self._load_best_source_model()
        classifier_path = Path(self.config.checkpoint_dir) / "fwc_classifier.pt"
        payload = torch.load(classifier_path, map_location=self.device, weights_only=False)
        self.model.classifier.load_state_dict(payload["classifier_state"])
        self.model.eval()
        target_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
        with torch.no_grad():
            for inputs, labels, record_ids, conditions in target_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                raw_features = self.model.extract_raw_features(inputs).detach().cpu().numpy()
                for index, record_id in enumerate(record_ids):
                    key = str(record_id)
                    entry = target_groups.setdefault(
                        key,
                        {"features": [], "label": int(labels[index]), "condition": str(conditions[index])},
                    )
                    entry["features"].append(raw_features[index])
        record_ids = []
        labels = []
        probabilities = []
        predictions = []
        diagnostics = []
        for record_id in sorted(target_groups):
            entry = target_groups[record_id]
            features = torch.as_tensor(
                np.stack(entry["features"]), dtype=torch.float32, device=self.device
            )
            started = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            ended = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            if started is not None:
                started.record()
            result = self.fwc.apply_collectibles(
                features,
                record_id,
                classifier=self.model.classifier,
                condition_id=None,
                source_report=self.pooled_report,
            )
            if ended is not None:
                ended.record()
                torch.cuda.synchronize(self.device)
                inference_ms = float(started.elapsed_time(ended))
            else:
                inference_ms = 0.0
            record_probabilities = torch.as_tensor(
                result.diagnostics["final_record_probabilities"], dtype=torch.float32
            ).numpy()
            record_ids.append(record_id)
            labels.append(int(entry["label"]))
            probabilities.append(record_probabilities)
            predictions.append(int(np.argmax(record_probabilities)))
            raw_selected = result.diagnostics.get("final_selected_item_details", [])
            decisions = result.diagnostics.get("decisions", [])
            diagnostics.append(
                {
                    "record_id": record_id,
                    "initial_layout": result.diagnostics["initial_layout"],
                    "final_layout": result.diagnostics["final_layout"],
                    "external_items": sorted(self.external_indices),
                    "red_priority": result.diagnostics["red_priority"],
                    "selected_items": raw_selected,
                    "collectibles": result.diagnostics.get("selected_collectibles", []),
                    "eviction_reasons": {
                        str(item["index"]): item.get("elimination_reason")
                        for item in decisions
                        if "elimination_reason" in item
                    },
                    "final_window_probabilities": result.diagnostics["final_window_probabilities"],
                    "record_prediction": int(np.argmax(record_probabilities)),
                    "inference_ms": inference_ms,
                }
            )
        return {
            "record_ids": np.asarray(record_ids, dtype=str),
            "labels": np.asarray(labels, dtype=np.int64),
            "probabilities": np.asarray(probabilities, dtype=float),
            "predictions": np.asarray(predictions, dtype=np.int64),
            "fwc_diagnostics": diagnostics,
        }

    def _load_best_source_model(self):
        path = Path(self.config.checkpoint_dir) / "best.pt"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        source_state = payload["model_state"]
        extractor_state = {
            key.removeprefix("extractor."): value
            for key, value in source_state.items()
            if key.startswith("extractor.")
        }
        classifier_state = {
            key.removeprefix("classifier."): value
            for key, value in source_state.items()
            if key.startswith("classifier.")
        }
        self.model.extractor.load_state_dict(extractor_state)
        self.model.classifier.load_state_dict(classifier_state)
        self.model.to(self.device)
        self.model.eval()

    def _evaluate_loader_identity(self, loader):
        groups = self._extract_grouped_features(loader, self.model, identity_logits=True)
        probabilities = []
        labels = []
        predictions = []
        with torch.no_grad():
            for record_id in sorted(groups):
                entry = groups[record_id]
                logits = torch.as_tensor(entry["logits"], device=self.device)
                mean_probability = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
                probabilities.append(mean_probability)
                labels.append(int(entry["label"]))
                predictions.append(int(np.argmax(mean_probability)))
        return compute_classification_metrics(
            np.asarray(labels), np.asarray(predictions), NUM_CLASSES
        )

    def _evaluate_grouped(self, groups, *, use_collectibles):
        labels, predictions = [], []
        for record_id in sorted(groups):
            entry = groups[record_id]
            features = torch.as_tensor(entry["features"], dtype=torch.float32, device=self.device)
            if use_collectibles:
                result = self.fwc.apply_collectibles(
                    features,
                    record_id,
                    classifier=self.model.classifier,
                    source_report=self.pooled_report,
                )
                probability = np.asarray(result.diagnostics["final_record_probabilities"])
            else:
                gate = self._cached_source_gate(features, record_id)
                probability = torch.softmax(
                    self.model.classifier(features * gate), dim=1
                ).mean(dim=0).detach().cpu().numpy()
            labels.append(int(entry["label"]))
            predictions.append(int(np.argmax(probability)))
        return compute_classification_metrics(np.asarray(labels), np.asarray(predictions), NUM_CLASSES)

    def _extract_grouped_features(self, loader, model, *, identity_logits=False):
        grouped: dict[str, dict[str, Any]] = {}
        model.eval()
        with torch.no_grad():
            for inputs, labels, record_ids, conditions in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                raw_features = model.extract_raw_features(inputs)
                logits = model.classifier(raw_features)
                raw_features = raw_features.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
                for index, record_id in enumerate(record_ids):
                    key = str(record_id)
                    entry = grouped.setdefault(
                        key,
                        {
                            "features": [],
                            "logits": [],
                            "label": int(labels[index]),
                            "condition": str(conditions[index]),
                        },
                    )
                    entry["features"].append(raw_features[index])
                    entry["logits"].append(logits[index])
        for entry in grouped.values():
            entry["features"] = np.stack(entry["features"])
            entry["logits"] = np.stack(entry["logits"])
        return grouped

    def _cached_source_gate(self, features, record_id):
        """Reuse a deterministic source-record gate across classifier epochs."""

        key = str(record_id)
        cached = self._source_gate_cache.get(key)
        if cached is None or cached.shape != features.shape or cached.device != features.device:
            cached = self.fwc.record_gate(features, record_id).window_gates.detach()
            self._source_gate_cache[key] = cached
        return cached

    @staticmethod
    def _mean_record_table(grouped):
        record_ids = sorted(grouped)
        return {
            "features": np.stack([grouped[key]["features"].mean(axis=0) for key in record_ids]),
            "labels": np.asarray([grouped[key]["label"] for key in record_ids], dtype=np.int64),
            "record_ids": np.asarray(record_ids, dtype=str),
            "condition_ids": np.asarray([grouped[key]["condition"] for key in record_ids], dtype=str),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
