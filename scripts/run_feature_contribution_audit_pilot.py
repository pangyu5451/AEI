"""Run a deterministic source-only contribution-audit pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from AGG_FWC.models import (  # noqa: E402
    audit_source_feature_contributions,
    save_feature_contribution_audit,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser


def _source_data():
    features = np.array(
        [
            [-1.0, 0.1], [-1.0, -0.2], [1.0, 0.0],
            [-1.0, 0.3], [-1.0, -0.1], [1.0, 0.2],
            [1.0, -0.2], [1.0, 0.1], [-1.0, 0.0],
            [1.0, 0.2], [1.0, -0.1], [-1.0, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    record_ids = np.repeat(np.array(["r2", "r1", "r4", "r3"]), 3)
    condition_ids = np.repeat(np.array(["z", "z", "a", "a"]), 3)
    return features, labels, record_ids, condition_ids


def _classifier():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-3.0, 0.0], [3.0, 0.0]]))
    return model


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed != 2026:
        raise ValueError("the pilot is fixed to seed 2026")
    run_dir = (args.output_root / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"pilot run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    features, labels, record_ids, condition_ids = _source_data()
    report = audit_source_feature_contributions(
        _classifier(),
        features,
        labels,
        record_ids,
        condition_ids,
        device=args.device,
        seed=args.seed,
        batch_size=4,
    )
    report_path = save_feature_contribution_audit(report, run_dir / "feature_contribution_audit")
    order = np.argsort(-np.asarray(report.global_nll_increase))
    summary = {
        "run_id": args.run_id,
        "seed": args.seed,
        "device": args.device,
        "source_only": report.source_only,
        "feature_count": report.feature_count,
        "condition_ids": list(report.condition_ids),
        "record_count_by_condition": list(report.record_count_by_condition),
        "top_features_by_global_nll_increase": [
            {
                "feature_index": int(index),
                "nll_increase": float(report.global_nll_increase[index]),
                "macro_f1_drop": float(report.global_macro_f1_drop[index]),
            }
            for index in order[:5]
        ],
        "report_npz": str(report_path),
        "report_json": str(report_path.with_suffix(".json")),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    metadata = {
        "command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "git_commit": commit,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

