"""Strict, record-preserving PU data materialization for FWC experiments.

The legacy AGG reader concatenates files before cutting windows and therefore
cannot be used for the staged FWC protocol.  This module keeps one manifest
row as one record, selects a fixed number of windows from that record, and
attaches auditable record metadata to the resulting window loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset

from AGG_FWC.datasets.strict_pu_adapter import build_strict_pu_loaders


SIGNAL_SIZE = 1024
RAW_WINDOWS_PER_RECORD = 100
FWC_WINDOWS_PER_RECORD = 8


class RecordWindowDataset(Dataset):
    """Materialize fixed windows while retaining record and condition IDs."""

    def __init__(
        self,
        records: pd.DataFrame,
        *,
        windows_per_record: int = FWC_WINDOWS_PER_RECORD,
        normalizetype: str = "mean-std",
        signal_reader: Callable[[Path, pd.Series], np.ndarray] | None = None,
    ) -> None:
        required = {
            "record_id",
            "condition_id",
            "fault_label",
            "class_index",
            "path",
        }
        missing = sorted(required.difference(records.columns))
        if missing:
            raise ValueError("record frame is missing: " + ", ".join(missing))
        if records.empty:
            raise ValueError("record frame must be non-empty")
        if isinstance(windows_per_record, bool) or not isinstance(windows_per_record, int):
            raise ValueError("windows_per_record must be a positive integer")
        if windows_per_record <= 0 or windows_per_record > RAW_WINDOWS_PER_RECORD:
            raise ValueError(
                f"windows_per_record must be in [1, {RAW_WINDOWS_PER_RECORD}]"
            )
        if normalizetype not in {"-1-1", "mean-std"}:
            raise ValueError("normalizetype must be '-1-1' or 'mean-std'")

        self.records = records.sort_values("record_id", kind="stable").reset_index(drop=True)
        self.windows_per_record = windows_per_record
        self.normalizetype = normalizetype
        self.signal_reader = _read_pu_signal if signal_reader is None else signal_reader
        self._windows: list[tuple[np.ndarray, int, str, str]] = []
        self._materialize()

    def _materialize(self) -> None:
        for _, row in self.records.iterrows():
            signal = np.asarray(self.signal_reader(Path(row["path"]), row), dtype=np.float32)
            signal = signal.reshape(-1)
            available_windows = signal.size // SIGNAL_SIZE
            if available_windows < self.windows_per_record:
                raise ValueError(
                    f"record {row['record_id']!r} contains fewer than "
                    f"{self.windows_per_record} complete windows"
                )
            signal = signal[: available_windows * SIGNAL_SIZE]
            selected = np.linspace(
                0, available_windows - 1, self.windows_per_record, dtype=np.int64
            )
            for window_number in selected.tolist():
                start = int(window_number) * SIGNAL_SIZE
                window = signal[start : start + SIGNAL_SIZE].copy()
                window = _normalize(window, self.normalizetype)
                self._windows.append(
                    (
                        window[None, :].astype(np.float32, copy=False),
                        int(row["class_index"]),
                        str(row["record_id"]),
                        str(row["condition_id"]),
                    )
                )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int):
        signal, label, record_id, condition_id = self._windows[index]
        return torch.from_numpy(signal), label, record_id, condition_id


def build_audited_window_loader(
    records: pd.DataFrame,
    *,
    partition: str,
    batch_size: int,
    windows_per_record: int = FWC_WINDOWS_PER_RECORD,
    normalizetype: str = "mean-std",
    num_workers: int = 0,
    signal_reader: Callable[[Path, pd.Series], np.ndarray] | None = None,
):
    """Build a real iterable loader with record-level audit metadata."""

    if partition not in {"source", "target"}:
        raise ValueError("partition must be 'source' or 'target'")
    dataset = RecordWindowDataset(
        records,
        windows_per_record=windows_per_record,
        normalizetype=normalizetype,
        signal_reader=signal_reader,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=partition == "source",
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    ordered = records.sort_values("record_id", kind="stable")
    loader.partition = partition
    loader.record_level = True
    loader.record_ids = tuple(str(value) for value in ordered["record_id"])
    loader.condition_ids = tuple(str(value) for value in ordered["condition_id"])
    loader.fault_labels = tuple(str(value) for value in ordered["fault_label"])
    loader.source_paths = tuple(str(Path(value).resolve()) for value in ordered["path"])
    return loader


def _read_pu_signal(path: Path, row: pd.Series) -> np.ndarray:
    """Read the vibration channel using the same nested PU field as AGG."""

    current_name = path.stem
    payload = scipy.io.loadmat(path)
    try:
        signal = payload[current_name][0][0][2][0][6][2]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"cannot read PU vibration channel from {path}") from exc
    return np.asarray(signal, dtype=np.float32).reshape(-1)[: RAW_WINDOWS_PER_RECORD * SIGNAL_SIZE]


def _normalize(window: np.ndarray, normalizetype: str) -> np.ndarray:
    if not np.isfinite(window).all():
        raise ValueError("raw window contains non-finite values")
    if normalizetype == "mean-std":
        mean = float(window.mean())
        std = float(window.std())
        return (window - mean) / (std if std > 0.0 else 1.0)
    centered = window - float(window.mean())
    lower, upper = float(centered.min()), float(centered.max())
    if upper <= lower:
        return np.zeros_like(window)
    return 2.0 * (centered - lower) / (upper - lower) - 1.0
