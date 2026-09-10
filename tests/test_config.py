from pathlib import Path

import pytest

from AGG_FWC.config import build_config, prepare_artifact_dirs


def test_build_config_uses_contract_defaults_and_scopes_output_dirs(tmp_path):
    config = build_config(output_root=tmp_path, run_id="test-contract")

    assert config.output_root == tmp_path
    assert config.data_name == "PU"
    assert config.method == "AGG"
    assert config.normalizetype == "mean-std"
    assert config.batch_size == 32
    assert config.lr == 1e-3
    assert config.epoch == 60
    assert config.seed == 2026

    directories = tuple(Path(directory).resolve() for directory in (
        config.checkpoint_dir,
        config.result_dir,
        config.log_dir,
    ))
    assert all(tmp_path.resolve() in directory.parents for directory in directories)
    assert len(set(directories)) == 3
    assert tuple(directory.parent.name for directory in directories) == (
        "checkpoints",
        "results",
        "logs",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"lr": 0},
        {"lr": -1},
        {"epoch": 0},
        {"num_workers": -1},
        {"normalizetype": "bad"},
        {"steps": "oops"},
        {"model_name": 123},
    ],
)
def test_build_config_rejects_invalid_overrides(tmp_path, overrides):
    with pytest.raises(ValueError):
        build_config(output_root=tmp_path, **overrides)


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "nested/child", "nested\\child", ".", "..", ""],
)
def test_build_config_rejects_run_id_path_components(tmp_path, run_id):
    with pytest.raises(ValueError):
        build_config(output_root=tmp_path, run_id=run_id)


def test_prepare_artifact_dirs_rolls_back_on_creation_failure(tmp_path, monkeypatch):
    config = build_config(output_root=tmp_path, run_id="rollback")
    original_mkdir = Path.mkdir

    def mkdir_with_injected_failure(self, *args, **kwargs):
        if self == config.result_dir:
            raise OSError("injected failure")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_with_injected_failure)

    with pytest.raises(OSError, match="injected failure"):
        prepare_artifact_dirs(config)

    assert not config.checkpoint_dir.exists()
    assert not config.result_dir.exists()
    assert not config.log_dir.exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "logs").exists()
