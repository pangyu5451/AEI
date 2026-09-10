"""Independent configuration for the AGG feature-wise-combination baseline.

This module only constructs and validates configuration values.  It does not
read data, create models, or start training.
"""

from dataclasses import dataclass, field
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import tempfile
from numbers import Real
from typing import Any


_ALLOWED_DATA_NAMES = {"PU", "PHM2009"}
_ALLOWED_METHODS = {"AGG"}
_ALLOWED_NORMALIZATIONS = {"-1-1", "mean-std"}
_ALLOWED_PROTOCOL_MODES = {"legacy", "strict"}
_POSITIVE_INT_FIELDS = (
    "epoch",
    "attribute_epoch",
    "attribute_hidden_dim",
    "batch_size",
    "num_train_samples",
    "num_test_samples",
    "in_channel",
)
_NON_NEGATIVE_INT_FIELDS = ("num_workers",)
_POSITIVE_NUMBER_FIELDS = ("lr", "inner_lr", "attribute_lr")
_NON_NEGATIVE_NUMBER_FIELDS = ("momentum", "weight_decay", "trade_off", "gamma")


@dataclass
class Config:
    """Runtime configuration and isolated artifact locations for one run."""

    model_name: str = "DG_for_RMFD"
    data_name: str = "PU"
    transfer_task: list[list[int] | int] = field(default_factory=lambda: [[0, 1, 2], 3])
    normalizetype: str = "mean-std"
    in_channel: int = 1
    num_train_samples: int = 200
    num_test_samples: int = 50
    cuda_device: str = "0"
    batch_size: int = 32
    num_workers: int = 0
    method: str = "AGG"
    lr: float = 1e-3
    inner_lr: float = 1e-3
    trade_off: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-5
    gamma: float = 0.1
    steps: str = "20, 40"
    epoch: int = 60
    fwc_stage: str = "baseline"
    protocol_mode: str = "legacy"
    attribute_hidden_dim: int = 128
    attribute_epoch: int = 1
    attribute_lr: float = 1e-3
    seed: int = 2026
    data_root: Path = Path("D:/datasets/PU")
    output_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    run_id: str = "default"
    smoke: bool = False
    require_cuda: bool = False
    checkpoint_dir: Path = field(init=False)
    result_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.output_root = Path(self.output_root)
            self.data_root = Path(self.data_root)
        except (TypeError, ValueError) as exc:
            raise ValueError("data_root and output_root must be valid paths") from exc
        _validate(self)
        self.checkpoint_dir, self.result_dir, self.log_dir = _artifact_dirs(
            self.output_root, self.run_id
        )


def _validate(config: Config) -> None:
    if not isinstance(config.model_name, str) or not config.model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(config.data_name, str) or config.data_name not in _ALLOWED_DATA_NAMES:
        raise ValueError(f"data_name must be one of {sorted(_ALLOWED_DATA_NAMES)}")
    if not isinstance(config.method, str) or config.method not in _ALLOWED_METHODS:
        raise ValueError("method must be 'AGG'")
    if not isinstance(config.smoke, bool):
        raise ValueError("smoke must be a boolean")
    if not isinstance(config.require_cuda, bool):
        raise ValueError("require_cuda must be a boolean")
    if not isinstance(config.fwc_stage, str) or not config.fwc_stage.strip():
        raise ValueError("fwc_stage must be a non-empty string")
    if not isinstance(config.protocol_mode, str) or config.protocol_mode not in _ALLOWED_PROTOCOL_MODES:
        raise ValueError(f"protocol_mode must be one of {sorted(_ALLOWED_PROTOCOL_MODES)}")
    if not isinstance(config.normalizetype, str) or config.normalizetype not in _ALLOWED_NORMALIZATIONS:
        raise ValueError(f"normalizetype must be one of {sorted(_ALLOWED_NORMALIZATIONS)}")
    for field_name in _POSITIVE_INT_FIELDS:
        _require_positive_int(field_name, getattr(config, field_name))
    for field_name in _NON_NEGATIVE_INT_FIELDS:
        _require_non_negative_int(field_name, getattr(config, field_name))
    for field_name in _POSITIVE_NUMBER_FIELDS:
        _require_number(field_name, getattr(config, field_name), positive=True)
    for field_name in _NON_NEGATIVE_NUMBER_FIELDS:
        _require_number(field_name, getattr(config, field_name), positive=False)
    if not isinstance(config.steps, str) or not config.steps.strip() or any(
        re.fullmatch(r"\d+", item.strip()) is None for item in config.steps.split(",")
    ):
        raise ValueError("steps must be a comma-separated string of non-negative integers")
    if (
        not isinstance(config.transfer_task, list)
        or len(config.transfer_task) != 2
        or not isinstance(config.transfer_task[0], list)
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in config.transfer_task[0])
        or len(config.transfer_task[0]) != 3
        or not isinstance(config.transfer_task[1], int)
        or isinstance(config.transfer_task[1], bool)
    ):
        raise ValueError("transfer_task must be [source_list, target_integer]")
    _validate_run_id(config.run_id)


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_number(name: str, value: Any, *, positive: bool) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (value <= 0) if positive else (value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_run_id(run_id: Any) -> None:
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).is_absolute()
    ):
        raise ValueError("run_id must be one non-empty relative path component")


def _artifact_dirs(output_root: Path, run_id: str) -> tuple[Path, Path, Path]:
    _validate_run_id(run_id)
    root = output_root.resolve()
    return (
        (root / "checkpoints" / run_id).resolve(),
        (root / "results" / run_id).resolve(),
        (root / "logs" / run_id).resolve(),
    )


def build_config(output_root=None, run_id="default", **overrides) -> Config:
    """Build one validated, isolated AGG configuration.

    ``output_root`` and ``run_id`` define the three run-specific artifact
    directories.  Other keyword arguments override dataclass defaults.
    """
    try:
        values = dict(overrides)
        values["output_root"] = (
            Path(output_root) if output_root is not None else Path(__file__).resolve().parent
        )
        values["run_id"] = run_id
        config = Config(**values)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"unknown or invalid configuration override: {exc}") from exc
    return config


def prepare_artifact_dirs(config: Config) -> tuple[Path, Path, Path]:
    """Create a run's checkpoint, result, and log directories exactly once."""
    directories = (config.checkpoint_dir, config.result_dir, config.log_dir)
    if any(directory.exists() for directory in directories):
        raise FileExistsError("one or more AGG run artifact directories already exist")
    category_dirs = tuple(directory.parent for directory in directories)
    category_existed = {directory: directory.exists() for directory in category_dirs}
    created: list[Path] = []
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=False)
            created.append(directory)
    except Exception:
        for directory in reversed(created):
            try:
                directory.rmdir()
            except OSError:
                pass
        for directory in reversed(category_dirs):
            if not category_existed[directory] and directory.exists():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        raise
    return directories


def save_config_snapshot(config: Config) -> Path:
    """Save a JSON-serializable snapshot without overwriting an existing file."""

    def jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    snapshot = config.result_dir / "config.json"
    payload = jsonable(asdict(config))
    if snapshot.exists():
        raise FileExistsError(snapshot)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{snapshot.name}.", suffix=".tmp", dir=snapshot.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, snapshot)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return snapshot
