import numpy as np
import pandas as pd
import pytest


def test_record_window_dataset_preserves_record_metadata_and_fixed_windows():
    from AGG_FWC.strict_pu_experiment import RecordWindowDataset

    frame = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "condition_id": "N15_M01_F10",
                "bearing_id": "KA04",
                "fault_label": "bearing_class:KA04",
                "class_index": 0,
                "path": "synthetic.mat",
            }
        ]
    )

    def read_signal(_path, _row):
        return np.arange(100 * 1024, dtype=np.float32)

    dataset = RecordWindowDataset(
        frame,
        windows_per_record=4,
        normalizetype="mean-std",
        signal_reader=read_signal,
    )

    assert len(dataset) == 4
    signal, label, record_id, condition_id = dataset[0]
    assert signal.shape == (1, 1024)
    assert label == 0
    assert record_id == "r1"
    assert condition_id == "N15_M01_F10"
    assert torch_is_finite(signal)


def test_strict_loader_exposes_unique_record_metadata():
    from AGG_FWC.strict_pu_experiment import build_audited_window_loader

    frame = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "condition_id": "N15_M01_F10",
                "bearing_id": "KA04",
                "fault_label": "bearing_class:KA04",
                "class_index": 0,
                "path": "synthetic.mat",
            }
        ]
    )

    loader = build_audited_window_loader(
        frame,
        partition="source",
        batch_size=2,
        windows_per_record=2,
        signal_reader=lambda _path, _row: np.ones(2 * 1024, dtype=np.float32),
    )

    assert loader.partition == "source"
    assert loader.record_level is True
    assert loader.record_ids == ("r1",)
    batch = next(iter(loader))
    assert batch[0].shape == (2, 1, 1024)
    assert tuple(batch[2]) == ("r1", "r1")


def torch_is_finite(value):
    import torch

    tensor = torch.as_tensor(value)
    return bool(torch.isfinite(tensor).all())


def test_runtime_exposes_source_only_stage_callbacks():
    from AGG_FWC.strict_pu_runtime import required_stage_names

    assert required_stage_names() == (
        "source_train_record_val",
        "calibrate_sources",
        "fit_attribute_predictor",
        "train_classifier_with_fwc_on_sources",
        "evaluate_target_once",
    )


def test_run_dependency_factory_uses_manifest_and_run_local_audit(tmp_path, monkeypatch):
    from AGG_FWC.config import build_config
    import AGG_FWC.run as run_module

    config = build_config(
        output_root=tmp_path,
        run_id="strict-fwc",
        protocol_mode="strict",
        fwc_stage="fwc",
    )
    captured = {}

    def fake_factory(received_config, manifest_path, audit_path):
        captured["config"] = received_config
        captured["manifest_path"] = manifest_path
        captured["audit_path"] = audit_path
        return {"sentinel": True}

    monkeypatch.setattr(
        "AGG_FWC.strict_pu_runtime.build_strict_pu_fwc_dependencies", fake_factory
    )
    dependencies = run_module.build_real_strict_dependencies(config)
    assert dependencies == {"sentinel": True}
    assert captured["config"] is config
    assert str(captured["manifest_path"]).endswith(
        "pu_condition_manifest_paper_condition.csv"
    )
    assert captured["audit_path"] == config.result_dir / "protocol_audit.json"
