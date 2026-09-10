import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from AGG_FWC.config import build_config, prepare_artifact_dirs, save_config_snapshot
from AGG_FWC.models.fwc_types import ConditionAttributeReport
from AGG_FWC.train import StrictStagedRunner


AGG_FWC_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DEPENDENCIES = {
    "source_train_loader",
    "source_val_loader",
    "target_loader",
    "num_classes",
    "source_split",
    "target_split",
    "protocol_audit",
    "component_hashes",
    "component_files",
    "stage_callbacks",
    "target_evaluator",
}


def _runtime_module():
    path = AGG_FWC_ROOT / "strict_pu_runtime.py"
    assert path.is_file(), "strict_pu_runtime.py has not been implemented"
    spec = importlib.util.spec_from_file_location("AGG_FWC.strict_pu_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path):
    data_root = tmp_path / "pu"
    data_root.mkdir()
    rows = []
    for split, condition, count in (
        ("source_train", "SOURCE", 14),
        ("source_val", "SOURCE", 14),
        ("target", "TARGET", 2),
    ):
        for index in range(count):
            path = data_root / f"{split}-{index}.mat"
            path.write_bytes(f"{split}-{index}".encode("ascii"))
            rows.append(
                {
                    "record_id": f"{split}-record-{index}",
                    "condition_id": condition,
                    "bearing_id": f"bearing-{index}",
                    "fault_label": f"fault-class-{index % 14}",
                    "label_provenance": "synthetic test metadata",
                    "path": str(path),
                    "split": split,
                    "class_index": index % 14,
                }
            )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return data_root, manifest_path, rows


def _report(condition_id):
    values = tuple([0.5] * 512)
    return ConditionAttributeReport(
        condition_id=condition_id,
        contribution=values,
        stability=values,
        redundancy=tuple([0.0] * 512),
        c_quantiles=(0.0, 1.0),
        delta_margin_quantiles=(0.0, 1.0),
    )


def _patch_synthetic_runtime(monkeypatch, runtime):
    from AGG_FWC.strict_pu_experiment import build_audited_window_loader

    def synthetic_loader(records, **kwargs):
        return build_audited_window_loader(
            records,
            signal_reader=lambda _path, row: np.tile(
                np.linspace(-1.0, 1.0, 1024, dtype=np.float32), 8
            )
            + float(row["class_index"]),
            **kwargs,
        )

    monkeypatch.setattr(runtime, "build_audited_window_loader", synthetic_loader)

    calibration_calls = []

    def fake_calibrate(_classifier, features, labels, record_ids, *, condition_id, **kwargs):
        assert "target" not in {str(value) for value in record_ids}
        assert len(features) == len(labels) == len(record_ids)
        calibration_calls.append(condition_id)
        return _report(condition_id)

    monkeypatch.setattr(runtime, "calibrate_condition_from_model", fake_calibrate)

    def fake_single_feature_accuracy(features, labels, record_ids, **kwargs):
        assert "target" not in {str(value) for value in record_ids}
        return np.full(features.shape[1], 0.5, dtype=float)

    monkeypatch.setattr(runtime, "fast_single_feature_grouped_oof_accuracies", fake_single_feature_accuracy)

    def fake_apply_collectibles(self, features, record_id, *, classifier, **kwargs):
        with torch.no_grad():
            probabilities = torch.softmax(classifier(features), dim=1)
        return SimpleNamespace(
            diagnostics={
                "initial_layout": {},
                "final_layout": {},
                "red_priority": {},
                "selected_collectibles": [],
                "decisions": [],
                "final_selected_item_details": [],
                "final_window_probabilities": probabilities.cpu().tolist(),
                "final_record_probabilities": probabilities.mean(dim=0).cpu().tolist(),
            }
        )

    monkeypatch.setattr(
        runtime.FWCFeatureCombination, "apply_collectibles", fake_apply_collectibles
    )

    def fake_record_gate(self, features, record_id):
        return SimpleNamespace(window_gates=torch.ones_like(features))

    monkeypatch.setattr(runtime.FWCFeatureCombination, "record_gate", fake_record_gate)

    def fake_fit(predictor, **kwargs):
        assert all("target" not in str(value) for value in kwargs["source_record_ids"])
        assert "target_features" not in kwargs
        predictor.eval()
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)
        return predictor

    monkeypatch.setattr(runtime, "fit_attribute_predictor", fake_fit)
    return calibration_calls


def _dependencies(tmp_path, monkeypatch):
    runtime = _runtime_module()
    data_root, manifest_path, rows = _manifest(tmp_path)
    config = build_config(
        output_root=tmp_path / "artifacts",
        run_id="strict-pu-test",
        data_root=data_root,
        protocol_mode="strict",
        epoch=1,
        attribute_epoch=1,
        batch_size=64,
        num_workers=0,
    )
    prepare_artifact_dirs(config)
    calibration_calls = _patch_synthetic_runtime(monkeypatch, runtime)
    dependencies = runtime.build_strict_pu_fwc_dependencies(
        config, manifest_path, tmp_path / "audit.json"
    )
    return runtime, config, dependencies, rows, calibration_calls


def test_strict_pu_dependencies_expose_all_five_stage_callbacks(tmp_path, monkeypatch):
    runtime, _config, dependencies, _rows, _calls = _dependencies(tmp_path, monkeypatch)

    assert REQUIRED_DEPENDENCIES.issubset(dependencies)
    assert set(dependencies["stage_callbacks"]) == set(StrictStagedRunner.STAGES[:-1])
    assert all(callable(callback) for callback in dependencies["stage_callbacks"].values())
    assert callable(dependencies["target_evaluator"])
    assert dependencies["num_classes"] == 14
    assert runtime.__file__


def test_source_stages_never_iterate_target_loader(tmp_path, monkeypatch):
    _runtime, config, dependencies, _rows, calibration_calls = _dependencies(
        tmp_path, monkeypatch
    )
    runner = StrictStagedRunner(config, **dependencies)

    runner.run_source_stages()

    assert runner.target_loader_iterations == 0
    assert calibration_calls == ["SOURCE"]
    assert runner.state["stage_order"] == list(StrictStagedRunner.STAGES[:-1])


def test_target_evaluator_emits_one_aggregated_row_per_record(tmp_path, monkeypatch):
    _runtime, config, dependencies, rows, _calls = _dependencies(tmp_path, monkeypatch)
    runner = StrictStagedRunner(config, **dependencies)
    runner.run_source_stages()
    save_config_snapshot(config)

    result = runner.evaluate_target_once()

    target_ids = [row["record_id"] for row in rows if row["split"] == "target"]
    assert tuple(result["record_predictions"]) == tuple(target_ids)
    assert runner.target_loader_iterations == 1
    with np.load(config.result_dir / "record_predictions.npz", allow_pickle=False) as archive:
        assert archive["probabilities"].shape == (len(target_ids), 14)
        assert archive["record_ids"].shape == (len(target_ids),)
    diagnostics = [
        json.loads(line)
        for line in (config.result_dir / "fwc_diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(diagnostics) == len(target_ids)
    assert {item["record_id"] for item in diagnostics} == set(target_ids)
    assert all("record_prediction" in item for item in diagnostics)


def test_component_paths_are_confined_to_agg_fwc(tmp_path, monkeypatch):
    _runtime, _config, dependencies, _rows, _calls = _dependencies(tmp_path, monkeypatch)

    code_components = {
        name: path
        for name, path in dependencies["component_files"].items()
        if name not in {"manifest", "protocol_audit"}
    }
    for path in code_components.values():
        resolved = Path(path).resolve()
        assert resolved == AGG_FWC_ROOT or AGG_FWC_ROOT in resolved.parents


def test_model_based_calibration_returns_source_report_without_target_inputs():
    runtime = _runtime_module()

    class TwoClassModel(torch.nn.Module):
        def forward(self, features):
            return torch.stack((features[:, 0], -features[:, 0]), dim=1)

    features = np.vstack(
        [
            np.column_stack((np.ones(10), np.zeros(10))),
            np.column_stack((-np.ones(10), np.zeros(10))),
        ]
    ).astype(np.float32)
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    record_ids = np.asarray([f"source-{index}" for index in range(20)])

    report = runtime.calibrate_condition_from_model(
        TwoClassModel(),
        features,
        labels,
        record_ids,
        condition_id="SOURCE",
        seed=2026,
        device=torch.device("cpu"),
    )

    assert report.condition_id == "SOURCE"
    contribution = np.asarray(report.contribution)
    assert contribution.shape == (2,)
    assert contribution[0] > contribution[1]
    assert np.isfinite(report.stability).all()


def test_fast_external_screening_keeps_single_feature_matrix_two_dimensional():
    runtime = _runtime_module()
    features = np.asarray(
        [[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    record_ids = np.asarray(["r0", "r1", "r2", "r3"])

    scores = runtime.fast_single_feature_grouped_oof_accuracies(
        features, labels, record_ids, seed=2026
    )

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_source_record_gate_is_cached_across_classifier_epochs():
    runtime = _runtime_module()

    class FakeFWC:
        def __init__(self):
            self.calls = 0

        def record_gate(self, features, record_id):
            self.calls += 1
            return SimpleNamespace(window_gates=torch.ones_like(features))

    experiment = object.__new__(runtime.StrictPUFWCRuntime)
    experiment.fwc = FakeFWC()
    experiment._source_gate_cache = {}
    features = torch.ones(2, 512)

    first = experiment._cached_source_gate(features, "r1")
    second = experiment._cached_source_gate(features, "r1")

    assert experiment.fwc.calls == 1
    assert torch.equal(first, second)
