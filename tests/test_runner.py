import ast
import hashlib
import inspect
import json
import logging
import os
import platform
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from AGG_FWC.config import (
    build_config,
    prepare_artifact_dirs,
    save_config_snapshot,
)
from AGG_FWC.run import build_parser, configure_cuda_device, set_seed
from AGG_FWC.train import AGGTrainer, collect_run_metadata, validate_dataset_lengths


def test_parser_defaults_to_agg_and_accepts_run_id():
    args = build_parser().parse_args(["--run_id", "unit"])

    assert args.method == "AGG"
    assert args.run_id == "unit"


def test_parser_rejects_non_agg_method():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--method", "DANN"])


def test_parser_accepts_epoch():
    args = build_parser().parse_args(["--epoch", "3"])

    assert args.epoch == 3


def test_parser_accepts_seed():
    args = build_parser().parse_args(["--seed", "2027"])

    assert args.seed == 2027


def test_pu_transfer_task_resolves_source_and_target_conditions():
    import AGG_FWC.run as run_module

    captured = {}

    def fake_manifest_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    def fake_dependency_builder(config, manifest_path, audit_path):
        captured["dependency_manifest_path"] = manifest_path
        captured["dependency_audit_path"] = audit_path
        return {}

    monkeypatch = pytest.MonkeyPatch()
    try:
        import AGG_FWC.datasets.build_pu_condition_manifest as manifest_module
        import AGG_FWC.strict_pu_runtime as runtime_module

        monkeypatch.setattr(manifest_module, "build_pu_condition_manifest", fake_manifest_builder)
        monkeypatch.setattr(runtime_module, "build_strict_pu_fwc_dependencies", fake_dependency_builder)
        config = SimpleNamespace(
            data_name="PU",
            transfer_task=[[0, 1, 2], 3],
            data_root=Path("data") / "PU",
            result_dir=Path("results") / "transfer-task-test",
        )
        run_module.build_real_strict_dependencies(config)
    finally:
        monkeypatch.undo()

    assert captured.get("source_conditions") == (
        "N09_M07_F10",
        "N15_M01_F10",
        "N15_M07_F04",
    )
    assert captured.get("target_condition") == "N15_M07_F10"


def test_parser_accepts_smoke_flag():
    args = build_parser().parse_args(["--smoke"])

    assert args.smoke is True


def test_parser_accepts_strict_protocol_mode():
    args = build_parser().parse_args(["--protocol-mode", "strict"])

    assert args.protocol_mode == "strict"


def test_parser_accepts_fwc_stage_selection():
    args = build_parser().parse_args(["--fwc-stage", "source-calibrated"])

    assert args.fwc_stage == "source-calibrated"


def test_cli_passes_smoke_state_into_config(monkeypatch, tmp_path):
    import AGG_FWC.run as run_module

    captured = {}

    def fake_build_config(**values):
        captured["values"] = values
        return build_config(**values)

    class FakeTrainer:
        def __init__(self, config):
            captured["config"] = config

        def train(self):
            captured["trained"] = True
            return []

    monkeypatch.setattr(run_module, "build_config", fake_build_config)
    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", FakeTrainer)

    run_module.main(
        [
            "--smoke",
            "--run_id",
            "smoke-cli",
            "--output_root",
            str(tmp_path),
        ]
    )

    assert captured["values"]["smoke"] is True
    assert captured["config"] is not None
    assert captured["trained"] is True


def test_cli_strict_mode_does_not_enter_legacy_trainer(tmp_path, monkeypatch):
    import AGG_FWC.run as run_module

    trainer_called = False

    class UnexpectedTrainer:
        def __init__(self, config):
            nonlocal trainer_called
            trainer_called = True

    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", UnexpectedTrainer)

    with pytest.raises(RuntimeError, match="StrictStagedRunner|injected"):
        run_module.main(
            [
                "--protocol-mode",
                "strict",
                "--run_id",
                "strict-cli",
                "--output_root",
                str(tmp_path),
            ]
        )

    assert trainer_called is False
    status = json.loads((tmp_path / "results" / "strict-cli" / "status.json").read_text())
    assert status["status"] == "blocked"
    assert status["legacy"] is False
    assert len(status["config_hash"]) == 64
    config_payload = json.loads((tmp_path / "results" / "strict-cli" / "config.json").read_text())
    assert config_payload["protocol_mode"] == "strict"


def test_legacy_status_marks_successful_legacy_run(tmp_path, monkeypatch):
    import AGG_FWC.run as run_module

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train(self):
            return []

    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", FakeTrainer)

    run_module.main(["--run_id", "legacy-status", "--output_root", str(tmp_path)])

    status = json.loads((tmp_path / "results" / "legacy-status" / "status.json").read_text())
    assert status["legacy"] is True


def test_run_metadata_contains_python_version_and_seed(tmp_path):
    config = build_config(output_root=tmp_path, run_id="metadata")
    metadata = collect_run_metadata(config)

    assert metadata["python_version"] == platform.python_version()
    assert metadata["seed"] == 2026
    assert metadata["require_cuda"] is False
    assert metadata["cuda_fallback"] is (not metadata["cuda_available"])


def test_config_snapshot_is_saved_as_json(tmp_path):
    config = build_config(output_root=tmp_path, run_id="snapshot")
    prepare_artifact_dirs(config)
    snapshot = save_config_snapshot(config)

    assert snapshot == config.result_dir / "config.json"
    assert snapshot.exists()


def test_config_snapshot_records_smoke_boolean(tmp_path):
    config = build_config(output_root=tmp_path, run_id="snapshot-smoke", smoke=True)
    prepare_artifact_dirs(config)

    snapshot = save_config_snapshot(config)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    assert payload["smoke"] is True


def test_cli_parser_accepts_required_cuda_flag():
    args = build_parser().parse_args(["--require-cuda"])

    assert args.require_cuda is True


def test_required_cuda_unavailable_writes_blocked_status_without_metrics(
    tmp_path, monkeypatch
):
    import AGG_FWC.run as run_module

    trainer_called = False

    class UnexpectedTrainer:
        def __init__(self, config):
            nonlocal trainer_called
            trainer_called = True

        def train(self):
            raise AssertionError("blocked CUDA run must not train")

    monkeypatch.setattr(run_module, "is_cuda_available", lambda: False)
    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", UnexpectedTrainer)

    run_module.main(
        [
            "--require-cuda",
            "--run_id",
            "cuda-blocked",
            "--output_root",
            str(tmp_path),
        ]
    )

    status_path = tmp_path / "results" / "cuda-blocked" / "status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["complete"] is False
    assert payload["formal_result"] is False
    assert "CUDA" in payload["reason"]
    assert trainer_called is False
    assert not (status_path.parent / "metrics.json").exists()
    assert "run blocked before training" in (
        tmp_path / "logs" / "cuda-blocked" / "train.log"
    ).read_text(encoding="utf-8")


def test_cli_writes_success_status_after_fake_training(tmp_path, monkeypatch):
    import AGG_FWC.run as run_module

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train(self):
            return []

    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", FakeTrainer)

    run_module.main(
        [
            "--smoke",
            "--run_id",
            "status-success",
            "--output_root",
            str(tmp_path),
        ]
    )

    status_path = tmp_path / "results" / "status-success" / "status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["complete"] is True
    assert payload["smoke"] is True
    assert payload["formal_result"] is False


def test_success_status_explicitly_records_cpu_fallback(tmp_path, monkeypatch):
    import AGG_FWC.run as run_module

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train(self):
            return []

    monkeypatch.setattr(run_module, "is_cuda_available", lambda: False)
    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", FakeTrainer)

    run_module.main(
        ["--run_id", "cpu-fallback", "--output_root", str(tmp_path)]
    )

    payload = json.loads(
        (tmp_path / "results" / "cpu-fallback" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["cuda_available"] is False
    assert payload["cuda_fallback"] is True
    assert payload["device"] == "cpu"


def test_training_failure_writes_incomplete_status_without_fake_metrics(
    tmp_path, monkeypatch
):
    import AGG_FWC.run as run_module

    class FailingTrainer:
        def __init__(self, config):
            self.config = config

        def train(self):
            raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("AGG_FWC.train.AGGTrainer", FailingTrainer)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_module.main(
            ["--run_id", "training-incomplete", "--output_root", str(tmp_path)]
        )

    result_dir = tmp_path / "results" / "training-incomplete"
    payload = json.loads((result_dir / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    assert payload["complete"] is False
    assert payload["formal_result"] is False
    assert "synthetic training failure" in payload["reason"]
    assert not (result_dir / "metrics.json").exists()
    assert "synthetic training failure" in (
        tmp_path / "logs" / "training-incomplete" / "train.log"
    ).read_text(encoding="utf-8")


def test_legacy_smoke_status_is_explicitly_legacy():
    status_path = Path(__file__).parents[1] / "results" / "task6-smoke-20260908" / "status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))

    assert payload["legacy"] is True
    assert payload["formal_result"] is False


def test_configure_cuda_device_sets_requested_visibility(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    configure_cuda_device("2")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_validate_dataset_lengths_rejects_empty_dataset():
    datasets = {"src_1": [], "src_2": [1], "src_3": [1], "tar": [1]}

    with pytest.raises(ValueError):
        validate_dataset_lengths(datasets)


def test_config_rejects_non_three_source_task(tmp_path):
    with pytest.raises(ValueError):
        build_config(output_root=tmp_path, transfer_task=[[0, 1], 3])


def _run_log_handler_present(config):
    expected = Path(config.log_dir / "train.log").resolve()
    return any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == expected
        for handler in logging.getLogger().handlers
    )


def test_initialization_exception_closes_run_log_handler(tmp_path, monkeypatch):
    config = build_config(output_root=tmp_path, run_id="init-error")
    prepare_artifact_dirs(config)

    def fail_loading(self):
        raise RuntimeError("injected dataset initialization failure")

    monkeypatch.setattr(AGGTrainer, "_load_datasets", fail_loading)

    with pytest.raises(RuntimeError, match="dataset initialization failure"):
        AGGTrainer(config)

    assert not _run_log_handler_present(config)


def test_legacy_trainer_rejects_strict_protocol_before_loading_target(tmp_path, monkeypatch):
    config = build_config(output_root=tmp_path, run_id="strict-legacy-guard", protocol_mode="strict")
    prepare_artifact_dirs(config)

    def unexpected_dataset_load(self):
        raise AssertionError("strict protocol must not enter the legacy dataset path")

    monkeypatch.setattr(AGGTrainer, "_load_datasets", unexpected_dataset_load)

    with pytest.raises(RuntimeError, match="StrictStagedRunner|strict"):
        AGGTrainer(config)


def test_training_exception_closes_run_log_handler(tmp_path):
    config = build_config(output_root=tmp_path, run_id="train-error", epoch=1)
    prepare_artifact_dirs(config)
    trainer = object.__new__(AGGTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.model = object()
    trainer.dataloaders = {}
    trainer._configure_logging()

    with pytest.raises(AttributeError):
        trainer.train()

    assert not _run_log_handler_present(config)


def test_training_source_has_no_batch_cpu_numpy_conversion():
    source = textwrap.dedent(inspect.getsource(AGGTrainer.train))
    tree = ast.parse(source)
    final_labels_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "final_labels"
            for target in node.targets
        )
    )
    numpy_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "numpy"
    ]

    assert numpy_lines
    assert all(line >= final_labels_line for line in numpy_lines)


def test_training_source_only_converts_to_cpu_numpy_after_final_target_collection():
    source = textwrap.dedent(inspect.getsource(AGGTrainer.train))
    tree = ast.parse(source)

    def is_cpu_numpy_call(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "numpy"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "cpu"
        )

    cpu_numpy_lines = [node.lineno for node in ast.walk(tree) if is_cpu_numpy_call(node)]
    assert cpu_numpy_lines

    final_collection_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "append"
            and isinstance(candidate.func.value, ast.Name)
            and candidate.func.value.id in {"final_labels", "final_predictions"}
            for candidate in ast.walk(node)
        )
    )
    assert all(line > final_collection_loop.end_lineno for line in cpu_numpy_lines)

    training_and_epoch_validation_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(candidate, ast.Subscript)
            and isinstance(candidate.value, ast.Attribute)
            and candidate.value.attr == "dataloaders"
            for candidate in ast.walk(node)
        )
        and not any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "append"
            and isinstance(candidate.func.value, ast.Name)
            and candidate.func.value.id in {"final_labels", "final_predictions"}
            for candidate in ast.walk(node)
        )
    ]
    assert training_and_epoch_validation_loops
    for loop in training_and_epoch_validation_loops:
        assert not any(loop.lineno <= line <= loop.end_lineno for line in cpu_numpy_lines)


def test_trainer_writes_all_artifacts_only_after_final_target_evaluation(
    tmp_path, monkeypatch
):
    import AGG_FWC.train as train_module

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(1, 2)

        def forward(self, inputs):
            return self.classifier(inputs.float().reshape(inputs.shape[0], -1))

    config = build_config(
        output_root=tmp_path,
        run_id="trainer-artifacts",
        epoch=1,
        batch_size=1,
        num_workers=0,
    )
    prepare_artifact_dirs(config)
    trainer = object.__new__(AGGTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.model = FakeModel()
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer.optimizer, milestones=[], gamma=0.1
    )
    batch = (torch.tensor([[1.0]]), torch.tensor([0]))
    trainer.dataloaders = {
        "src_1": [batch],
        "src_2": [batch],
        "src_3": [batch],
        "tar": [batch],
    }
    trainer.datasets = {"num_cls": 2}
    trainer._configure_logging()
    events = []

    original_compute = train_module.compute_classification_metrics
    original_save_metrics = train_module.save_metrics_json
    original_savez = train_module.np.savez
    original_write_text = Path.write_text
    original_torch_save = torch.save
    import AGG_FWC.config as config_module

    original_save_config = config_module.save_config_snapshot

    def compute_and_mark(*args, **kwargs):
        metrics = original_compute(*args, **kwargs)
        events.append("final_evaluation_finished")
        return metrics

    def save_metrics_and_mark(*args, **kwargs):
        events.append("metrics.json")
        return original_save_metrics(*args, **kwargs)

    def savez_and_mark(file, *args, **kwargs):
        if Path(file).name == "predictions.npz":
            events.append("predictions.npz")
        return original_savez(file, *args, **kwargs)

    def write_text_and_mark(self, data, *args, **kwargs):
        if self.name == "epoch_metrics.json":
            events.append("epoch_metrics.json")
        return original_write_text(self, data, *args, **kwargs)

    def torch_save_and_mark(obj, file, *args, **kwargs):
        if Path(file).name == "final.pt":
            events.append("final.pt")
        return original_torch_save(obj, file, *args, **kwargs)

    def save_config_and_mark(config_to_save):
        events.append("config.json")
        return original_save_config(config_to_save)

    monkeypatch.setattr(train_module, "compute_classification_metrics", compute_and_mark)
    monkeypatch.setattr(train_module, "save_metrics_json", save_metrics_and_mark)
    monkeypatch.setattr(train_module.np, "savez", savez_and_mark)
    monkeypatch.setattr(Path, "write_text", write_text_and_mark)
    monkeypatch.setattr(torch, "save", torch_save_and_mark)
    monkeypatch.setattr(config_module, "save_config_snapshot", save_config_and_mark)

    trainer.train()

    required_files = (
        config.result_dir / "config.json",
        config.result_dir / "predictions.npz",
        config.result_dir / "metrics.json",
        config.result_dir / "epoch_metrics.json",
        config.checkpoint_dir / "final.pt",
    )
    assert all(path.exists() for path in required_files)
    assert set(events) == {
        "final_evaluation_finished",
        "config.json",
        "predictions.npz",
        "metrics.json",
        "epoch_metrics.json",
        "final.pt",
    }
    evaluation_index = events.index("final_evaluation_finished")
    assert all(events.index(name) > evaluation_index for name in required_files_names())
    assert [events.index(name) for name in required_files_names()] == sorted(
        events.index(name) for name in required_files_names()
    )

    payload = json.loads((config.result_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == config.run_id
    assert payload["config_path"] == str(config.result_dir / "config.json")
    assert payload["checkpoint_path"] == str(config.checkpoint_dir / "final.pt")


def required_files_names():
    return ("config.json", "final.pt", "predictions.npz", "epoch_metrics.json", "metrics.json")


def test_parser_rejects_non_three_source_task():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--transfer_task", "[[0,1],3]"])


def _strict_config(tmp_path, run_id):
    config = build_config(
        output_root=tmp_path,
        run_id=run_id,
        protocol_mode="strict",
        epoch=1,
        attribute_epoch=1,
        batch_size=1,
        num_workers=0,
    )
    prepare_artifact_dirs(config)
    return config


class _ExplodingTargetLoader:
    def __init__(self):
        self.partition = "target"
        self.record_level = True
        self.record_ids = ["target-record"]
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        raise AssertionError("target loader must not be touched during source stages")


def _strict_runner_or_fail(*args, **kwargs):
    import AGG_FWC.train as train_module

    assert hasattr(train_module, "StrictStagedRunner"), (
        "Task7 must expose StrictStagedRunner"
    )
    kwargs.setdefault("protocol_mode", "strict")
    return train_module.StrictStagedRunner(*args, **kwargs)


def test_strict_source_stages_never_touch_target_loader(tmp_path):
    target_loader = _ExplodingTargetLoader()
    calls = []
    config = _strict_config(tmp_path, "strict-source-only")
    component_hashes, component_files = _component_hashes(tmp_path)

    def source_epoch_stage(source_train_loader, source_val_loader, epoch, state):
        calls.append((source_train_loader, source_val_loader, epoch))
        checkpoint = config.checkpoint_dir / f"epoch-{epoch}.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"checkpoint_path": checkpoint, "val_metric": 0.5, "frozen": True}

    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", rows=[("train", 0)]),
        source_val_loader=_PartitionedLoader("source", rows=[("val", 0)]),
        target_loader=target_loader,
        num_classes=2,
        source_split={"train": ["source-train"], "val": ["source-val"]},
        target_split=["target-test"],
        protocol_audit=_valid_protocol_audit(["train"], ["val"], ["target-record"]),
        stage_callbacks={
            "source_train_record_val": source_epoch_stage,
            "calibrate_sources": lambda source_train_loader, source_val_loader, state: {
                "calibration": "frozen", "frozen": True
            },
            "fit_attribute_predictor": lambda source_train_loader, source_val_loader, state: {
                "attribute_predictor": "frozen", "frozen": True
            },
            "train_classifier_with_fwc_on_sources": lambda source_train_loader, source_val_loader, state: {
                "classifier": "frozen", "frozen": True
            },
        },
        component_hashes=component_hashes,
        component_files=component_files,
    )

    runner.run_source_stages()

    assert len(calls) == config.epoch
    assert target_loader.iterations == 0


def test_strict_final_evaluation_requires_frozen_checkpoint(tmp_path):
    config = _strict_config(tmp_path, "strict-checkpoint-gate")
    component_hashes, component_files = _component_hashes(tmp_path)
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
        num_classes=2,
        source_split={"train": ["source-train"], "val": ["source-val"]},
        target_split=["target-test"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
    )

    with pytest.raises(RuntimeError, match="checkpoint"):
        runner.evaluate_target_once()


def test_strict_final_evaluation_iterates_target_once_and_writes_record_artifacts(
    tmp_path,
):
    class CountingTargetLoader:
        def __init__(self):
            self.partition = "target"
            self.record_level = True
            self.record_ids = ["target-record-1", "target-record-2"]
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            yield ("target-record-1", 0, [0.9, 0.1])
            yield ("target-record-2", 1, [0.2, 0.8])

    target_loader = CountingTargetLoader()
    config = _strict_config(tmp_path, "strict-final-once")
    component_hashes, component_files = _component_hashes(tmp_path)

    def checkpoint_stage(source_train_loader, source_val_loader, epoch, state):
        checkpoint = config.checkpoint_dir / f"strict-final-{epoch}.pt"
        checkpoint.write_bytes(b"frozen-checkpoint")
        return {"checkpoint_path": checkpoint, "val_metric": 1.0, "frozen": True}

    def target_evaluator(target_loader, state):
        rows = list(target_loader)
        return {
            "record_ids": [row[0] for row in rows],
            "labels": [row[1] for row in rows],
            "probabilities": [row[2] for row in rows],
            "predictions": [int(max(range(2), key=row[2].__getitem__)) for row in rows],
            "fwc_diagnostics": [
                {
                    "record_id": row[0],
                    "initial_layout": {"selected_indices": []},
                    "final_layout": {"selected_indices": []},
                    "external_items": [],
                    "red_priority": {"decision": "value_optimum"},
                    "selected_items": [
                        {"raw_utility": 0.1, "effective_utility": 0.2, "multiplier": 2.0}
                    ],
                    "collectibles": [],
                    "eviction_reasons": [],
                    "final_window_probabilities": [row[2]],
                    "record_prediction": int(max(range(2), key=row[2].__getitem__)),
                    "candidates": [],
                    "inference_ms": 0.25,
                }
                for row in rows
            ],
        }

    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", rows=[("train", 0)]),
        source_val_loader=_PartitionedLoader("source", rows=[("val", 0)]),
        target_loader=target_loader,
        num_classes=2,
        source_split={"train": ["source-train"], "val": ["source-val"]},
        target_split=["target-test"],
        protocol_audit=_valid_protocol_audit(
            ["train"], ["val"], ["target-record-1", "target-record-2"]
        ),
        stage_callbacks={
            "source_train_record_val": checkpoint_stage,
            "calibrate_sources": lambda source_train_loader, source_val_loader, state: {
                "calibration": "frozen",
                "frozen": True,
            },
            "fit_attribute_predictor": lambda source_train_loader, source_val_loader, state: {
                "attribute_predictor": "frozen",
                "frozen": True,
            },
            "train_classifier_with_fwc_on_sources": lambda source_train_loader, source_val_loader, state: {
                "classifier": "frozen",
                "frozen": True,
            },
        },
        target_evaluator=target_evaluator,
        component_hashes=component_hashes,
        component_files=component_files,
    )

    result = runner.run()

    assert target_loader.iterations == 1
    assert result["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert (config.result_dir / "record_predictions.npz").exists()
    assert (config.result_dir / "fwc_diagnostics.jsonl").exists()
    with np.load(config.result_dir / "record_predictions.npz", allow_pickle=False) as payload:
        assert payload["record_ids"].tolist() == ["target-record-1", "target-record-2"]
        assert payload["labels"].tolist() == [0, 1]
        assert payload["predictions"].tolist() == [0, 1]
        assert payload["probabilities"].shape == (2, 2)

    with (config.result_dir / "fwc_diagnostics.jsonl").open(encoding="utf-8") as handle:
        diagnostics = [json.loads(line) for line in handle]
    assert diagnostics[0]["initial_layout"] == {"selected_indices": []}
    assert diagnostics[0]["inference_ms"] == pytest.approx(0.25)

    config_payload = json.loads((config.result_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["protocol_mode"] == "strict"
    assert config_payload["seed"] == config.seed
    manifest = json.loads((config.result_dir / "strict_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_split"] == {"train": ["source-train"], "val": ["source-val"]}
    assert manifest["target_split"] == ["target-test"]
    assert manifest["fwc_stage"] == config.fwc_stage
    assert manifest["component_hashes"] == component_hashes
    assert manifest["component_files"] == {
        name: str(path.resolve()) for name, path in component_files.items()
    }
    assert manifest["stages"] == [
        "source_train_record_val",
        "calibrate_sources",
        "fit_attribute_predictor",
        "train_classifier_with_fwc_on_sources",
        "evaluate_target_once",
    ]
    assert manifest["target_loader_iterations"] == 1
    assert manifest["target_evaluation_count"] == 1
    assert len(manifest["config_hash"]) == 64
    assert len(manifest["protocol_audit_hash"]) == 64
    assert len(manifest["checkpoint_hash"]) == 64
    assert manifest["best_epoch"] == 0
    assert manifest["best_val_metric"] == pytest.approx(1.0)

    with pytest.raises(RuntimeError, match="once"):
        runner.evaluate_target_once()


def test_strict_finalization_rejects_unfrozen_source_attribute(tmp_path):
    config = _strict_config(tmp_path, "strict-frozen-gate")
    component_hashes, component_files = _component_hashes(tmp_path)
    checkpoint = config.checkpoint_dir / "strict-final.pt"

    def checkpoint_stage(source_train_loader, source_val_loader, epoch, state):
        checkpoint.write_bytes(b"frozen-checkpoint")
        return {"checkpoint_path": checkpoint, "val_metric": 0.5, "frozen": True}

    callbacks = {
        "source_train_record_val": checkpoint_stage,
        "calibrate_sources": lambda source_train_loader, source_val_loader, state: {
            "calibration": "frozen",
            "frozen": True,
        },
        "fit_attribute_predictor": lambda source_train_loader, source_val_loader, state: {
            "attribute_predictor": "unfrozen",
            "frozen": False,
        },
        "train_classifier_with_fwc_on_sources": lambda source_train_loader, source_val_loader, state: {
            "classifier": "frozen",
            "frozen": True,
        },
    }
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["source-train"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["source-val"]),
        target_loader=_PartitionedLoader("target", record_ids=["target-test"]),
        num_classes=2,
        source_split={"train": ["source-train"], "val": ["source-val"]},
        target_split=["target-test"],
        protocol_audit=_valid_protocol_audit(
            ["source-train"], ["source-val"], ["target-test"]
        ),
        stage_callbacks=callbacks,
        target_evaluator=lambda target_loader, state: {},
        component_hashes=component_hashes,
        component_files=component_files,
    )

    runner.run_source_stages()
    save_config_snapshot(config)

    with pytest.raises(RuntimeError, match="frozen"):
        runner.evaluate_target_once()


def test_strict_stages_have_explicit_source_and_final_stage_names():
    import AGG_FWC.train as train_module

    assert tuple(getattr(train_module.StrictStagedRunner, "STAGES", ())) == (
        "source_train_record_val",
        "calibrate_sources",
        "fit_attribute_predictor",
        "train_classifier_with_fwc_on_sources",
        "evaluate_target_once",
    )


class _PartitionedLoader:
    def __init__(self, partition, record_level=True, rows=(), record_ids=None):
        self.partition = partition
        self.record_level = record_level
        self.rows = list(rows)
        inferred = [row[0] for row in self.rows if row]
        self.record_ids = list(
            record_ids if record_ids is not None else (inferred or [f"{partition}-{id(self)}"])
        )
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.rows)


def _valid_protocol_audit(train_ids=None, val_ids=None, target_ids=None):
    train_ids = list(train_ids or ["record-a"])
    val_ids = list(val_ids or ["record-b"])
    target_ids = list(target_ids or ["record-c"])
    splits = {"train": train_ids, "val": val_ids, "test": target_ids}
    canonical_records = sorted(
        train_ids + val_ids + target_ids, key=lambda value: str(value)
    )
    split_hashes = {
        name: hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for name, values in splits.items()
    }
    manifest_hash = hashlib.sha256(
        json.dumps(canonical_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audit_source = Path(__file__).resolve()
    file_hash = hashlib.sha256(audit_source.read_bytes()).hexdigest()
    return {
        "status": "passed",
        "record_count": len(canonical_records),
        "fault_class_counts": {"normal": 3},
        "bearing_intersections": {"train__val": [], "train__test": [], "val__test": []},
        "file_hashes": {str(audit_source): file_hash},
        "manifest_hash": manifest_hash,
        "split_hashes": split_hashes,
        "splits": splits,
    }


def _component_hashes(tmp_path):
    component = tmp_path / "component.bin"
    component.write_bytes(b"component")
    digest = hashlib.sha256(component.read_bytes()).hexdigest()
    return {"model": digest}, {"model": component}


def test_strict_runner_rejects_protocol_audit_loader_record_id_mismatch(tmp_path):
    config = _strict_config(tmp_path, "strict-audit-loader-binding")
    component_hashes, component_files = _component_hashes(tmp_path)

    with pytest.raises(RuntimeError, match="audit.*record|record.*audit|mismatch"):
        _strict_runner_or_fail(
            config,
            source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
            source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
            target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
            num_classes=2,
            source_split={"train": ["train-r"], "val": ["val-r"]},
            target_split=["target-r"],
            protocol_audit=_valid_protocol_audit(
                ["audit-train"], ["audit-val"], ["audit-target"]
            ),
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_runner_rejects_forged_protocol_file_hash(tmp_path):
    config = _strict_config(tmp_path, "strict-audit-file-hash")
    component_hashes, component_files = _component_hashes(tmp_path)
    audit = _valid_protocol_audit(["train-r"], ["val-r"], ["target-r"])
    audit["file_hashes"] = {next(iter(audit["file_hashes"])): "a" * 64}

    with pytest.raises(RuntimeError, match="file hash|hash.*mismatch|audit"):
        _strict_runner_or_fail(
            config,
            source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
            source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
            target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
            num_classes=2,
            source_split={"train": ["train-r"], "val": ["val-r"]},
            target_split=["target-r"],
            protocol_audit=audit,
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_runner_rejects_source_loader_without_source_record_metadata(tmp_path):
    config = _strict_config(tmp_path, "strict-loader-metadata")
    component_hashes, component_files = _component_hashes(tmp_path)

    with pytest.raises(ValueError, match="partition.*source|record_level"):
        _strict_runner_or_fail(
            config,
            source_train_loader=[("train", 0)],
            source_val_loader=_PartitionedLoader("source"),
            target_loader=_PartitionedLoader("target"),
            num_classes=2,
            source_split={"train": ["source-train"], "val": ["source-val"]},
            target_split=["target-test"],
            protocol_audit=_valid_protocol_audit(),
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_runner_rejects_target_partition_as_source_validation(tmp_path):
    config = _strict_config(tmp_path, "strict-target-val")
    component_hashes, component_files = _component_hashes(tmp_path)

    with pytest.raises(ValueError, match="target|source"):
        _strict_runner_or_fail(
            config,
            source_train_loader=_PartitionedLoader("source", rows=[("train", 0)]),
            source_val_loader=_PartitionedLoader("target", rows=[("val", 0)]),
            target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
            num_classes=2,
            source_split={"train": ["source-train"], "val": ["source-val"]},
            target_split=["target-test"],
            protocol_audit=_valid_protocol_audit(["train"], ["val"], ["target-r"]),
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_runner_selects_best_checkpoint_from_source_validation_each_epoch(tmp_path):
    config = _strict_config(tmp_path, "strict-best-source-epoch")
    config.epoch = 3
    source_train = _PartitionedLoader("source", rows=[("train", 0)])
    source_val = _PartitionedLoader("source", rows=[("val", 0)])
    target = _PartitionedLoader("target", record_ids=["target-r"])
    component_hashes, component_files = _component_hashes(tmp_path)
    metrics = [0.2, 0.9, 0.4]
    seen_epochs = []

    def source_epoch(source_train_loader, source_val_loader, epoch, state):
        seen_epochs.append((source_train_loader, source_val_loader, epoch))
        checkpoint = config.checkpoint_dir / f"epoch-{epoch}.pt"
        checkpoint.write_bytes(f"epoch-{epoch}".encode("ascii"))
        return {
            "checkpoint_path": checkpoint,
            "val_metric": metrics[epoch],
            "frozen": True,
        }

    runner = _strict_runner_or_fail(
        config,
        source_train_loader=source_train,
        source_val_loader=source_val,
        target_loader=target,
        num_classes=2,
        source_split={"train": ["source-train"], "val": ["source-val"]},
        target_split=["target-test"],
        protocol_audit=_valid_protocol_audit(["train"], ["val"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
        stage_callbacks={
            "source_train_record_val": source_epoch,
            "calibrate_sources": lambda train, val, state: {"frozen": True},
            "fit_attribute_predictor": lambda train, val, state: {
                "attribute_predictor": "source", "frozen": True
            },
            "train_classifier_with_fwc_on_sources": lambda train, val, state: {
                "classifier": "source", "frozen": True
            },
        },
    )

    runner.run_source_stages()

    assert [epoch for _, _, epoch in seen_epochs] == [0, 1, 2]
    assert all(train is source_train and val is source_val for train, val, _ in seen_epochs)
    assert runner.state["stage_results"]["source_train_record_val"]["best_epoch"] == 1
    assert runner.state["stage_results"]["source_train_record_val"]["best_val_metric"] == pytest.approx(0.9)
    assert (config.checkpoint_dir / "best.pt").read_bytes() == b"epoch-1"


def test_strict_runner_rejects_non_sha256_component_hash(tmp_path):
    config = _strict_config(tmp_path, "strict-hash-format")
    source = _PartitionedLoader("source")
    target = _PartitionedLoader("target")

    with pytest.raises(ValueError, match="sha256|64"):
        _strict_runner_or_fail(
            config,
            source_train_loader=source,
            source_val_loader=source,
            target_loader=target,
            num_classes=2,
            source_split={"train": ["source-train"], "val": ["source-val"]},
            target_split=["target-test"],
            protocol_audit=_valid_protocol_audit(),
            component_hashes={"model": "not-a-hash"},
        )


def test_strict_runner_exposes_required_diagnostic_schema():
    import AGG_FWC.train as train_module

    assert set(train_module.StrictStagedRunner.REQUIRED_DIAGNOSTIC_FIELDS) == {
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
    }


def test_strict_runner_rejects_incomplete_nonfinite_and_non_json_diagnostics(tmp_path):
    config = _strict_config(tmp_path, "strict-diagnostic-validation")
    component_hashes, component_files = _component_hashes(tmp_path)
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
        num_classes=2,
        source_split={"train": ["source-train"], "val": ["source-val"]},
        target_split=["target-test"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
    )
    complete = {
        "record_id": "r1",
        "initial_layout": {},
        "final_layout": {},
        "external_items": [],
        "red_priority": {},
        "selected_items": [],
        "collectibles": [],
        "eviction_reasons": [],
        "final_window_probabilities": [[0.5, 0.5]],
        "record_prediction": 0,
        "inference_ms": 0.5,
    }

    for diagnostic, message in (
        ({"record_id": "r1"}, "missing"),
        ({**complete, "inference_ms": float("nan")}, "finite"),
        ({**complete, "red_priority": object()}, "JSON"),
    ):
        with pytest.raises(ValueError, match=message):
            runner._save_fwc_diagnostics(
                {"fwc_diagnostics": [diagnostic]}, np.asarray(["r1"]), 1.0
            )


def test_strict_state_machine_rejects_out_of_order_stage_without_mutation(tmp_path):
    config = _strict_config(tmp_path, "strict-state-order")
    component_hashes, component_files = _component_hashes(tmp_path)
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
        num_classes=2,
        source_split={"train": ["train-r"], "val": ["val-r"]},
        target_split=["target-r"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
        stage_callbacks={
            "calibrate_sources": lambda train, val, state: {"frozen": True},
        },
    )

    with pytest.raises(RuntimeError, match="order|source_train_record_val"):
        runner.calibrate_sources()

    assert runner.state["stage_order"] == []


def test_early_strict_evaluation_failure_is_retryable_after_source_prerequisites(tmp_path):
    config = _strict_config(tmp_path, "strict-evaluate-retry")
    component_hashes, component_files = _component_hashes(tmp_path)
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=_PartitionedLoader("target", rows=[("target-r", 0)], record_ids=["target-r"]),
        num_classes=1,
        source_split={"train": ["train-r"], "val": ["val-r"]},
        target_split=["target-r"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
    )

    with pytest.raises(RuntimeError, match="checkpoint"):
        runner.evaluate_target_once()

    assert runner.state["stage_order"] == []
    assert runner.target_loader_iterations == 0
    assert runner._target_evaluation_started is False


def test_strict_runner_rejects_failed_or_incomplete_protocol_audit(tmp_path):
    config = _strict_config(tmp_path, "strict-audit-gate")
    component_hashes, component_files = _component_hashes(tmp_path)
    audit = _valid_protocol_audit()
    audit["status"] = "blocked"

    with pytest.raises(RuntimeError, match="passed|audit"):
        _strict_runner_or_fail(
            config,
            source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
            source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
            target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
            num_classes=2,
            source_split={"train": ["train-r"], "val": ["val-r"]},
            target_split=["target-r"],
            protocol_audit=audit,
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_runner_rejects_overlapping_source_record_ids(tmp_path):
    config = _strict_config(tmp_path, "strict-record-overlap")
    component_hashes, component_files = _component_hashes(tmp_path)

    with pytest.raises((ValueError, RuntimeError), match="overlap|record"):
        _strict_runner_or_fail(
            config,
            source_train_loader=_PartitionedLoader("source", record_ids=["same-r"]),
            source_val_loader=_PartitionedLoader("source", record_ids=["same-r"]),
            target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
            num_classes=2,
            source_split={"train": ["same-r"], "val": ["same-r"]},
            target_split=["target-r"],
            protocol_audit=_valid_protocol_audit(["same-r"], ["same-r"], ["target-r"]),
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_target_result_rejects_float_labels_and_incomplete_records(tmp_path):
    config = _strict_config(tmp_path, "strict-target-contract")
    component_hashes, component_files = _component_hashes(tmp_path)
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=_PartitionedLoader("target", record_ids=["target-r1", "target-r2"]),
        num_classes=2,
        source_split={"train": ["train-r"], "val": ["val-r"]},
        target_split=["target-r1", "target-r2"],
        protocol_audit=_valid_protocol_audit(
            ["train-r"], ["val-r"], ["target-r1", "target-r2"]
        ),
        component_hashes=component_hashes,
        component_files=component_files,
    )

    with pytest.raises(ValueError, match="integer"):
        runner._validate_target_result(
            {
                "record_ids": ["target-r1", "target-r2"],
                "labels": [0.0, 1.0],
                "probabilities": [[0.9, 0.1], [0.1, 0.9]],
                "predictions": [0, 1],
            }
        )
    with pytest.raises(ValueError, match="target|record"):
        runner._validate_target_result(
            {
                "record_ids": ["target-r1"],
                "labels": [0],
                "probabilities": [[0.9, 0.1]],
                "predictions": [0],
            }
        )


def test_strict_hash_contract_exposes_content_verified_component_files(tmp_path):
    import inspect
    import AGG_FWC.train as train_module

    parameters = inspect.signature(train_module.StrictStagedRunner).parameters
    assert "component_files" in parameters


def test_strict_runner_rejects_component_file_tampering(tmp_path):
    config = _strict_config(tmp_path, "strict-hash-tampering")
    component_hashes, component_files = _component_hashes(tmp_path)
    component_files["model"].write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="mismatch"):
        _strict_runner_or_fail(
            config,
            source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
            source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
            target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
            num_classes=2,
            source_split={"train": ["train-r"], "val": ["val-r"]},
            target_split=["target-r"],
            protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
            component_hashes=component_hashes,
            component_files=component_files,
        )


def test_strict_runner_rejects_duplicate_stage_without_mutating_order(tmp_path):
    config = _strict_config(tmp_path, "strict-duplicate-stage")
    component_hashes, component_files = _component_hashes(tmp_path)

    def source_epoch(train, val, epoch, state):
        checkpoint = config.checkpoint_dir / f"epoch-{epoch}.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"checkpoint_path": checkpoint, "val_metric": 1.0, "frozen": True}

    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=_PartitionedLoader("target", record_ids=["target-r"]),
        num_classes=2,
        source_split={"train": ["train-r"], "val": ["val-r"]},
        target_split=["target-r"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
        stage_callbacks={
            "source_train_record_val": source_epoch,
            "calibrate_sources": lambda train, val, state: {"frozen": True},
        },
    )
    runner.source_train_record_val()
    runner.calibrate_sources()

    with pytest.raises(RuntimeError, match="order|expected"):
        runner.calibrate_sources()
    assert runner.state["stage_order"] == [
        "source_train_record_val",
        "calibrate_sources",
    ]


def test_early_evaluation_failure_can_retry_after_prerequisites(tmp_path):
    config = _strict_config(tmp_path, "strict-retry-after-early-failure")
    component_hashes, component_files = _component_hashes(tmp_path)
    target = _PartitionedLoader("target", rows=[("target-r", 0)], record_ids=["target-r"])

    def source_epoch(train, val, epoch, state):
        checkpoint = config.checkpoint_dir / f"epoch-{epoch}.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"checkpoint_path": checkpoint, "val_metric": 1.0, "frozen": True}

    def evaluator(loader, state):
        rows = list(loader)
        return {
            "record_ids": [row[0] for row in rows],
            "labels": np.asarray([row[1] for row in rows], dtype=np.int64),
            "probabilities": [[1.0] for _ in rows],
            "predictions": np.asarray([0 for _ in rows], dtype=np.int64),
            "fwc_diagnostics": [
                {
                    "record_id": row[0],
                    "initial_layout": {}, "final_layout": {}, "external_items": [],
                    "red_priority": {}, "selected_items": [], "collectibles": [],
                    "eviction_reasons": [], "final_window_probabilities": [[1.0]],
                    "record_prediction": 0, "inference_ms": 0.1,
                }
                for row in rows
            ],
        }

    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=target,
        num_classes=1,
        source_split={"train": ["train-r"], "val": ["val-r"]},
        target_split=["target-r"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
        stage_callbacks={
            "source_train_record_val": source_epoch,
            "calibrate_sources": lambda train, val, state: {"frozen": True},
            "fit_attribute_predictor": lambda train, val, state: {
                "attribute_predictor": "source", "frozen": True
            },
            "train_classifier_with_fwc_on_sources": lambda train, val, state: {
                "classifier": "source", "frozen": True
            },
        },
    )
    with pytest.raises(RuntimeError, match="config snapshot|checkpoint"):
        runner.evaluate_target_once()
    assert runner.state["stage_order"] == []
    runner.target_evaluator = evaluator
    runner.run_source_stages()
    save_config_snapshot(config)
    result = runner.evaluate_target_once()
    assert result["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert runner.target_loader_iterations == 1


def test_strict_run_publishes_no_partial_target_artifacts_when_diagnostics_write_fails(
    tmp_path, monkeypatch
):
    import AGG_FWC.train as train_module

    config = _strict_config(tmp_path, "strict-atomic-failure")
    component_hashes, component_files = _component_hashes(tmp_path)
    target = _PartitionedLoader("target", rows=[("target-r", 0)], record_ids=["target-r"])

    def source_epoch(train, val, epoch, state):
        checkpoint = config.checkpoint_dir / f"epoch-{epoch}.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"checkpoint_path": checkpoint, "val_metric": 1.0, "frozen": True}

    def evaluator(loader, state):
        list(loader)
        return {
            "record_ids": ["target-r"], "labels": np.asarray([0], dtype=np.int64),
            "probabilities": [[1.0]], "predictions": np.asarray([0], dtype=np.int64),
            "fwc_diagnostics": [{
                "record_id": "target-r", "initial_layout": {}, "final_layout": {},
                "external_items": [], "red_priority": {}, "selected_items": [],
                "collectibles": [], "eviction_reasons": [],
                "final_window_probabilities": [[1.0]], "record_prediction": 0,
                "inference_ms": 0.1,
            }],
        }

    real_atomic_write = train_module._atomic_write_text

    def fail_diagnostics_write(path, text, *, overwrite=True):
        if Path(path).name == "fwc_diagnostics.jsonl":
            raise OSError("injected diagnostics artifact failure")
        return real_atomic_write(path, text, overwrite=overwrite)

    monkeypatch.setattr(train_module, "_atomic_write_text", fail_diagnostics_write)
    runner = _strict_runner_or_fail(
        config,
        source_train_loader=_PartitionedLoader("source", record_ids=["train-r"]),
        source_val_loader=_PartitionedLoader("source", record_ids=["val-r"]),
        target_loader=target,
        num_classes=1,
        source_split={"train": ["train-r"], "val": ["val-r"]}, target_split=["target-r"],
        protocol_audit=_valid_protocol_audit(["train-r"], ["val-r"], ["target-r"]),
        component_hashes=component_hashes,
        component_files=component_files,
        stage_callbacks={
            "source_train_record_val": source_epoch,
            "calibrate_sources": lambda train, val, state: {"frozen": True},
            "fit_attribute_predictor": lambda train, val, state: {"attribute_predictor": "source", "frozen": True},
            "train_classifier_with_fwc_on_sources": lambda train, val, state: {"classifier": "source", "frozen": True},
        }, target_evaluator=evaluator,
    )
    with pytest.raises(OSError, match="diagnostics artifact"):
        runner.run()
    manifest = json.loads((config.result_dir / "strict_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "evaluate_target_once" not in manifest["stages"]
    assert not (config.result_dir / "record_predictions.npz").exists()
    assert not (config.result_dir / "fwc_diagnostics.jsonl").exists()
    assert not (config.result_dir / "metrics.json").exists()
    assert list(config.result_dir.glob(".strict-final-*")) == []


def test_strict_writers_expose_atomic_failure_path():
    import AGG_FWC.evaluate as evaluate_module
    import AGG_FWC.train as train_module

    assert hasattr(evaluate_module, "_atomic_write_text")
    assert hasattr(train_module, "_atomic_write_text")
