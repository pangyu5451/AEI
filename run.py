"""Independent command-line entry point for the AGG-only experiment."""

import argparse
import ast
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from AGG_FWC.config import build_config, prepare_artifact_dirs, save_config_snapshot


PU_CONDITION_INDEX_TO_ID = {
    0: "N09_M07_F10",
    1: "N15_M01_F10",
    2: "N15_M07_F04",
    3: "N15_M07_F10",
}


def parse_transfer_task(value):
    try:
        task = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "transfer_task must be a Python-style list"
        ) from exc
    if not isinstance(task, list) or len(task) != 2:
        raise argparse.ArgumentTypeError(
            "transfer_task must contain source and target conditions"
        )
    source, target = task
    if not isinstance(source, list) or not all(isinstance(item, int) for item in source):
        raise argparse.ArgumentTypeError("source conditions must be a list of integers")
    if len(source) != 3:
        raise argparse.ArgumentTypeError("transfer_task must contain exactly three source conditions")
    if isinstance(target, list) and len(target) == 1:
        target = target[0]
    if not isinstance(target, int):
        raise argparse.ArgumentTypeError("target condition must be an integer")
    return [source, target]


def resolve_pu_transfer_conditions(transfer_task):
    """Resolve the four PU condition indices used by the original dataset code."""

    try:
        source_indices, target_index = transfer_task
        source_conditions = tuple(PU_CONDITION_INDEX_TO_ID[index] for index in source_indices)
        target_condition = PU_CONDITION_INDEX_TO_ID[target_index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "PU transfer_task must contain known condition indices 0, 1, 2, or 3"
        ) from exc
    if len(set(source_conditions)) != len(source_conditions):
        raise ValueError("PU source conditions must be unique")
    if target_condition in source_conditions:
        raise ValueError("PU source and target conditions must be disjoint")
    return source_conditions, target_condition


def build_parser():
    parser = argparse.ArgumentParser(description="Train the independent AGG baseline")
    parser.add_argument("--model_name", default="DG_for_RMFD")
    parser.add_argument("--data_name", choices=["PHM2009", "PU"], default="PU")
    parser.add_argument("--transfer_task", type=parse_transfer_task, default=[[0, 1, 2], 3])
    parser.add_argument("--normalizetype", choices=["-1-1", "mean-std"], default="mean-std")
    parser.add_argument("--in_channel", type=int, default=1)
    parser.add_argument("--num_train_samples", type=int, default=200)
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--method", choices=["AGG"], default="AGG")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--trade_off", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--steps", default="20, 40")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--protocol-mode", choices=["legacy", "strict"], default="legacy")
    parser.add_argument("--fwc-stage", dest="fwc_stage", default="baseline")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run_id", default="default")
    parser.add_argument("--data_root", type=Path, default=Path("D:/datasets/PU"))
    parser.add_argument("--output_root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true")
    return parser


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_cuda_device(cuda_device):
    """Set the requested CUDA visibility before constructing a run."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip()


def is_cuda_available():
    import torch

    return torch.cuda.is_available()


def _append_run_log(config, message):
    log_path = Path(config.log_dir) / "train.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_run_status(config, status, reason=None, *, cuda_available=None, config_hash=None):
    if status not in {"success", "blocked", "incomplete"}:
        raise ValueError("status must be success, blocked, or incomplete")
    status_path = Path(config.result_dir) / "status.json"
    payload = {
        "status": status,
        "smoke": config.smoke,
        "complete": status == "success",
        "formal_result": status == "success" and not config.smoke,
        "legacy": config.protocol_mode == "legacy",
    }
    if reason:
        payload["reason"] = str(reason)
    if cuda_available is not None:
        payload["cuda_available"] = bool(cuda_available)
        payload["cuda_fallback"] = not cuda_available and not config.require_cuda
        payload["device"] = "cuda" if cuda_available else "cpu"
    if config_hash is not None:
        payload["config_hash"] = str(config_hash)
    if status_path.exists():
        raise FileExistsError(status_path)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{status_path.name}.", suffix=".tmp", dir=status_path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, status_path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return status_path


def _ensure_config_snapshot(config):
    snapshot = Path(config.result_dir) / "config.json"
    if not snapshot.exists():
        save_config_snapshot(config)
    return snapshot


class StrictRunnerDependenciesMissing(RuntimeError):
    """Raised when strict CLI execution has no real staged dependencies."""


def build_strict_runner(config, dependencies=None):
    """Construct strict runner only from explicit injected data/components."""
    required = {
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
    if not isinstance(dependencies, dict) or not required.issubset(dependencies):
        missing = sorted(required.difference(dependencies or {}))
        raise StrictRunnerDependenciesMissing(
            "strict CLI requires explicit injected staged dependencies; "
            f"missing {', '.join(missing)}"
        )
    from AGG_FWC.train import StrictStagedRunner

    return StrictStagedRunner(config, **dependencies)


def build_real_strict_dependencies(config):
    """Build the real PU FWC dependencies for an explicit FWC run."""

    if config.data_name != "PU":
        raise ValueError("real strict dependencies currently support only data_name='PU'")

    source_conditions, target_condition = resolve_pu_transfer_conditions(
        config.transfer_task
    )
    manifest_path = Path(config.result_dir) / "pu_condition_manifest.csv"
    audit_path = Path(config.result_dir) / "protocol_audit.json"
    from AGG_FWC.datasets.build_pu_condition_manifest import build_pu_condition_manifest

    build_pu_condition_manifest(
        data_root=config.data_root,
        manifest_path=manifest_path,
        audit_path=audit_path,
        source_conditions=source_conditions,
        target_condition=target_condition,
        build_loader_bundle=False,
    )
    from AGG_FWC.strict_pu_runtime import build_strict_pu_fwc_dependencies

    return build_strict_pu_fwc_dependencies(config, manifest_path, audit_path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_cuda_device(args.cuda_device)
    set_seed(args.seed)

    if args.smoke:
        args.epoch = min(args.epoch, 1)
        args.num_train_samples = min(args.num_train_samples, 2)
        args.num_test_samples = min(args.num_test_samples, 2)
        args.batch_size = min(args.batch_size, 8)

    config_values = vars(args).copy()
    config = build_config(**config_values)
    prepare_artifact_dirs(config)
    cuda_available = is_cuda_available()
    if config.require_cuda and not cuda_available:
        snapshot = _ensure_config_snapshot(config)
        message = (
            "CUDA is required by this configuration but is unavailable; "
            "run blocked before training"
        )
        _append_run_log(config, message)
        write_run_status(
            config,
            "blocked",
            message,
            cuda_available=cuda_available,
            config_hash=_file_hash(snapshot) if config.protocol_mode == "strict" else None,
        )
        return None
    if config.protocol_mode == "strict":
        try:
            dependencies = (
                build_real_strict_dependencies(config)
                if config.fwc_stage == "fwc"
                else None
            )
            runner = build_strict_runner(config, dependencies)
        except StrictRunnerDependenciesMissing as exc:
            message = str(exc)
            snapshot = _ensure_config_snapshot(config)
            _append_run_log(config, message)
            write_run_status(
                config,
                "blocked",
                message,
                cuda_available=cuda_available,
                config_hash=_file_hash(snapshot),
            )
            raise
        try:
            result = runner.run()
        except Exception as exc:
            _ensure_config_snapshot(config)
            message = f"strict training incomplete: {type(exc).__name__}: {exc}"
            _append_run_log(config, message)
            write_run_status(
                config, "incomplete", message, cuda_available=cuda_available
            )
            raise
        write_run_status(config, "success", "strict training completed", cuda_available=cuda_available)
        return result
    from AGG_FWC.train import AGGTrainer

    try:
        trainer = AGGTrainer(config)
        result = trainer.train()
    except Exception as exc:
        _ensure_config_snapshot(config)
        message = f"training incomplete: {type(exc).__name__}: {exc}"
        _append_run_log(config, message)
        write_run_status(
            config, "incomplete", message, cuda_available=cuda_available
        )
        raise
    _ensure_config_snapshot(config)
    write_run_status(config, "success", "training completed", cuda_available=cuda_available)
    return result


if __name__ == "__main__":
    main()
