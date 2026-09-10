"""Strict, metadata-first adapter for the legacy PU bearing reader.

The upstream :mod:`PU_bearing` reader is a window-level convenience loader:
it concatenates files within a condition and fault class before cutting fixed
windows, then returns only ``(window, fault_label)``.  That output does not
contain a trustworthy bearing-level identity.  This module therefore never
infers or fabricates ``record_id`` values from window order or file names.

The strict builder consumes a separately supplied, verified manifest with one
row per raw measurement record.  It validates the three-way split and writes
the same protocol audit consumed by ``StrictStagedRunner``.  It returns
record-level *loader metadata*, not a fake iterable dataset.  A later data
materialization layer must preserve the manifest record identity while
reading each record.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from AGG_FWC.protocol import (
    ProtocolBlockedError,
    validate_manifest,
    write_protocol_audit,
)


STRICT_SPLITS = ("source_train", "source_val", "target")
_AUDIT_SPLITS = {
    "source_train": "train",
    "source_val": "val",
    "target": "test",
}


@dataclass(frozen=True)
class RecordLevelLoaderMetadata:
    """Auditable identity metadata attached to one strict data partition.

    This class intentionally has no ``__iter__`` method.  It cannot be
    accidentally passed to a training loop as if it were a record-preserving
    tensor loader.  A real loader implementation may carry these attributes
    when it is added after the raw-record reader is verified.
    """

    partition: str
    record_level: bool
    record_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    bearing_ids: tuple[str, ...]
    fault_labels: tuple[str, ...]
    source_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.partition not in {"source", "target"}:
            raise ValueError("partition must be 'source' or 'target'")
        if self.record_level is not True:
            raise ValueError("strict metadata must be record_level=True")
        if not self.record_ids:
            raise ValueError("record_ids must be non-empty")
        lengths = {
            len(self.record_ids),
            len(self.condition_ids),
            len(self.bearing_ids),
            len(self.fault_labels),
            len(self.source_paths),
        }
        if len(lengths) != 1:
            raise ValueError("record metadata columns must have equal lengths")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique")

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata for logs and protocol manifests."""

        return {
            "partition": self.partition,
            "record_level": self.record_level,
            "record_ids": list(self.record_ids),
            "condition_ids": list(self.condition_ids),
            "bearing_ids": list(self.bearing_ids),
            "fault_labels": list(self.fault_labels),
            "source_paths": list(self.source_paths),
        }


@dataclass(frozen=True)
class StrictPULoaderBundle:
    """Three strict partition metadata objects plus the protocol audit."""

    source_train: RecordLevelLoaderMetadata
    source_val: RecordLevelLoaderMetadata
    target: RecordLevelLoaderMetadata
    manifest_path: Path
    data_root: Path
    audit_path: Path
    protocol_audit: Mapping[str, Any]
    protocol_mode: str
    bearing_disjoint: bool

    @property
    def loader_metadata(self) -> dict[str, RecordLevelLoaderMetadata]:
        return {
            "source_train": self.source_train,
            "source_val": self.source_val,
            "target": self.target,
        }

    def audit_payload(self) -> dict[str, Any]:
        """Return a JSON-safe bundle summary for run metadata."""

        return {
            "manifest_path": str(self.manifest_path),
            "data_root": str(self.data_root),
            "audit_path": str(self.audit_path),
            "protocol_mode": self.protocol_mode,
            "bearing_disjoint": self.bearing_disjoint,
            "protocol_audit": dict(self.protocol_audit),
            "loader_metadata": {
                name: metadata.as_dict()
                for name, metadata in self.loader_metadata.items()
            },
        }


def build_strict_pu_loaders(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    bearing_disjoint: bool | None = None,
    protocol_mode: str | None = None,
) -> StrictPULoaderBundle:
    """Validate a record manifest and build strict partition metadata.

    Parameters
    ----------
    data_root:
        Existing directory containing the raw PU archive.  Manifest paths
        must resolve below this directory.
    manifest_path:
        CSV or JSON manifest with one row per raw measurement record.  In
        addition to the protocol columns, it must contain ``split`` with
        exactly ``source_train``, ``source_val`` and ``target`` values.
    audit_path:
        New JSON path for the deterministic protocol audit.  Existing files
        are not overwritten.
    bearing_disjoint:
        Explicitly set to ``False`` for the paper's condition-transfer
        protocol.  Omitting it preserves the historical ``True`` behavior.
    protocol_mode:
        Optional explicit mode, either ``bearing_disjoint`` or
        ``paper_condition``.  If supplied, it must agree with
        ``bearing_disjoint``; passing ``paper_condition`` alone is an explicit
        opt-in to condition-based splitting.

    Raises
    ------
    ProtocolBlockedError
        If the archive, manifest, bearing metadata, split assignment, or
        source paths cannot support a leakage-safe record-level protocol.
    """

    root = _require_directory(data_root, "data_root")
    manifest_file = _require_file(manifest_path, "manifest")
    manifest = _read_manifest(manifest_file)
    validate_manifest(manifest)
    if "split" not in manifest.columns:
        raise ProtocolBlockedError(
            "manifest is missing split metadata required for source_train/source_val/target"
        )

    normalized_split = manifest["split"].map(_text)
    if normalized_split.isnull().any() or (normalized_split == "").any():
        raise ProtocolBlockedError("manifest split values must be non-blank")
    observed_splits = set(normalized_split)
    if observed_splits != set(STRICT_SPLITS):
        raise ProtocolBlockedError(
            "manifest split must contain exactly source_train, source_val, and target"
        )

    _assert_paths_below_root(manifest, root)
    split_frames = {
        audit_name: manifest.loc[normalized_split == source_name].copy()
        for source_name, audit_name in _AUDIT_SPLITS.items()
    }
    audit_file = Path(audit_path).expanduser().resolve()
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    write_protocol_audit(
        manifest,
        split_frames,
        audit_file,
        bearing_disjoint=bearing_disjoint,
        protocol_mode=protocol_mode,
    )
    protocol_audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if protocol_audit.get("status") != "passed":
        raise ProtocolBlockedError("protocol audit did not pass")

    return StrictPULoaderBundle(
        source_train=_metadata_for_frame(split_frames["train"], "source"),
        source_val=_metadata_for_frame(split_frames["val"], "source"),
        target=_metadata_for_frame(split_frames["test"], "target"),
        manifest_path=manifest_file,
        data_root=root,
        audit_path=audit_file,
        protocol_audit=protocol_audit,
        protocol_mode=protocol_audit["protocol_mode"],
        bearing_disjoint=protocol_audit["bearing_disjoint"],
    )


def load_legacy_pu_windows(
    *,
    data_root: str | Path,
    condition_index: int,
    num_samples: int,
    bearing_metadata: pd.DataFrame | None,
):
    """Read legacy PU windows only when bearing metadata is supplied.

    The returned value is exactly the upstream ``data_load`` output and is
    *not* record-level.  ``bearing_metadata=None`` is blocked explicitly so a
    caller cannot silently promote condition/class windows to a strict
    bearing-independent loader.  The strict builder above is the only path
    that creates auditable partition metadata.
    """

    _require_directory(data_root, "data_root")
    if bearing_metadata is None:
        raise ProtocolBlockedError(
            "legacy PU loader lacks bearing-level metadata; cannot fabricate record_id"
        )
    if not isinstance(bearing_metadata, pd.DataFrame):
        raise ProtocolBlockedError("bearing-level metadata must be a pandas DataFrame")
    required = {"record_id", "bearing_id"}
    missing = sorted(required.difference(bearing_metadata.columns))
    if missing:
        raise ProtocolBlockedError(
            "bearing-level metadata is missing: " + ", ".join(missing)
        )

    # Import lazily: importing the legacy module also imports optional
    # torchvision components, which are not needed for metadata validation.
    from .PU_bearing import data_load

    return data_load(str(Path(data_root).expanduser().resolve()), condition_index, num_samples)


def _read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("records")
            if not isinstance(payload, list):
                raise ProtocolBlockedError("JSON manifest must contain a list of records")
            return pd.DataFrame(payload)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ProtocolBlockedError(f"could not read manifest: {path}") from exc
    raise ProtocolBlockedError("manifest must be a .csv or .json file")


def _metadata_for_frame(frame: pd.DataFrame, partition: str) -> RecordLevelLoaderMetadata:
    ordered = frame.sort_values("record_id", kind="stable")
    return RecordLevelLoaderMetadata(
        partition=partition,
        record_level=True,
        record_ids=tuple(_text(value) for value in ordered["record_id"]),
        condition_ids=tuple(_text(value) for value in ordered["condition_id"]),
        bearing_ids=tuple(_text(value) for value in ordered["bearing_id"]),
        fault_labels=tuple(_text(value) for value in ordered["fault_label"]),
        source_paths=tuple(str(Path(value).expanduser().resolve()) for value in ordered["path"]),
    )


def _require_directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ProtocolBlockedError(f"{name} does not exist or is not a directory: {path}")
    return path


def _require_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ProtocolBlockedError(f"{name} does not exist or is not a file: {path}")
    return path


def _assert_paths_below_root(manifest: pd.DataFrame, root: Path) -> None:
    for value in manifest["path"]:
        # Check the lexical path against the logical data root.  PU archives
        # may use directory links for class folders; ``is_file`` still
        # verifies that the linked source exists, while protocol.py resolves
        # it for canonical hashing.
        source_path = Path(value).expanduser().absolute()
        try:
            source_path.relative_to(root.absolute())
        except ValueError as exc:
            raise ProtocolBlockedError(
                f"manifest source path is outside data_root: {source_path}"
            ) from exc
        if not source_path.is_file():
            raise ProtocolBlockedError(
                f"manifest source path does not exist: {source_path}"
            )


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
