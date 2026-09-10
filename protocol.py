"""Strict metadata validation and audit helpers for bearing experiments."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from AGG_FWC.models.fwc_types import _create_audited_source_partition


REQUIRED_MANIFEST_COLUMNS = (
    "record_id",
    "condition_id",
    "bearing_id",
    "fault_label",
    "label_provenance",
    "path",
)


class ProtocolBlockedError(RuntimeError):
    """Raised when metadata cannot support a leakage-safe experiment."""


DEFAULT_PROTOCOL_MODE = "bearing_disjoint"
PAPER_CONDITION_PROTOCOL_MODE = "paper_condition"
_PROTOCOL_MODES = {DEFAULT_PROTOCOL_MODE, PAPER_CONDITION_PROTOCOL_MODE}


def create_audited_source_partition(manifest, source_condition_ids, target_condition_ids):
    """Return an audited source/target partition for trusted-runner leak checks.

    This is not a security boundary against hostile in-process callers.  It
    records the manifest partition so Task3 helpers can catch accidental input
    mixing before optimizer mutation; Task7 enforces target-loader exclusion.
    """
    validate_manifest(manifest)
    source_conditions = _normalized_condition_ids(source_condition_ids, "source")
    target_conditions = _normalized_condition_ids(target_condition_ids, "target")
    if set(source_conditions).intersection(target_conditions):
        raise ProtocolBlockedError("source and target conditions must be disjoint")
    normalized_conditions = _normalized_column(manifest, "condition_id")
    declared = set(source_conditions).union(target_conditions)
    observed = set(normalized_conditions)
    if observed != declared:
        raise ProtocolBlockedError("source and target conditions must allocate the manifest exactly once")
    source_records = tuple(
        _normalized_column(manifest.loc[normalized_conditions.isin(source_conditions)], "record_id")
    )
    target_records = tuple(
        _normalized_column(manifest.loc[normalized_conditions.isin(target_conditions)], "record_id")
    )
    return _create_audited_source_partition(
        source_records, source_conditions, target_records, target_conditions
    )


def validate_manifest(manifest):
    """Return ``manifest`` after enforcing the metadata-only label contract."""
    if not isinstance(manifest, pd.DataFrame):
        raise ProtocolBlockedError("manifest must be a pandas DataFrame")
    if manifest.empty:
        raise ProtocolBlockedError("manifest must be non-empty")

    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS).difference(manifest.columns))
    if missing:
        raise ProtocolBlockedError(
            "manifest is missing required columns: " + ", ".join(missing)
        )

    values = manifest.loc[:, REQUIRED_MANIFEST_COLUMNS]
    if values.isnull().any().any():
        raise ProtocolBlockedError("manifest required columns must not contain missing values")

    normalized = values.apply(lambda column: column.map(_normalized_text))
    if (normalized == "").any().any():
        raise ProtocolBlockedError("manifest required columns must not contain blank values")
    if normalized["record_id"].duplicated().any():
        raise ProtocolBlockedError("manifest record_id values must be unique")
    canonical_paths = values["path"].map(_canonical_path)
    if canonical_paths.duplicated().any():
        raise ProtocolBlockedError("manifest path values must be unique")
    if (normalized["fault_label"] == normalized["bearing_id"]).any():
        raise ProtocolBlockedError(
            "fault_label must be verified metadata distinct from bearing_id; "
            "file-name label inference is blocked"
        )
    return manifest


def assert_disjoint_bearings(
    train, val, test, *, bearing_disjoint=True
):
    """Validate bearing intersections, blocking them by default.

    ``bearing_disjoint=False`` is an explicit opt-in for the original paper's
    condition-transfer protocol.  It records the intersections for callers
    but does not treat condition-based separation as bearing-independent
    generalization.
    """
    if not isinstance(bearing_disjoint, bool):
        raise ProtocolBlockedError("bearing_disjoint must be a boolean")
    frames = {"train": train, "val": val, "test": test}
    bearing_sets = {
        name: set(_normalized_column(validate_manifest(frame), "bearing_id"))
        for name, frame in frames.items()
    }
    overlaps = _bearing_intersections(bearing_sets)
    non_empty = {name: values for name, values in overlaps.items() if values}
    if non_empty and bearing_disjoint:
        details = "; ".join(
            f"{name}: {', '.join(values)}" for name, values in non_empty.items()
        )
        raise ProtocolBlockedError(f"bearing overlap across splits: {details}")
    return overlaps


def write_protocol_audit(
    manifest,
    split_frames,
    path,
    *,
    bearing_disjoint=None,
    protocol_mode=None,
):
    """Write a deterministic audit for a declared evaluation protocol.

    Omitting both protocol arguments preserves the historical strict default:
    ``bearing_disjoint=True`` and ``protocol_mode='bearing_disjoint'``.  The
    condition-transfer mode must be explicit and is the only mode that may
    contain cross-split bearing intersections.
    """
    bearing_disjoint, protocol_mode = _resolve_protocol_semantics(
        bearing_disjoint, protocol_mode
    )
    validate_manifest(manifest)
    if not isinstance(split_frames, dict) or not split_frames:
        raise ProtocolBlockedError("split_frames must be a non-empty dictionary")

    split_frames = dict(split_frames)
    if set(split_frames) != {"train", "val", "test"}:
        raise ProtocolBlockedError("split_frames must include train, val, and test")
    normalized_splits = {}
    for name, frame in split_frames.items():
        if not isinstance(name, str) or not name.strip():
            raise ProtocolBlockedError("split names must be non-empty strings")
        validate_manifest(frame)
        normalized_splits[name] = _canonical_records(frame)

    _validate_split_allocation(manifest, normalized_splits)
    bearing_sets = {
        name: {record["bearing_id"] for record in records}
        for name, records in normalized_splits.items()
    }
    intersections = _bearing_intersections(bearing_sets)
    if any(intersections.values()) and bearing_disjoint:
        details = "; ".join(
            f"{name}: {', '.join(values)}"
            for name, values in intersections.items()
            if values
        )
        raise ProtocolBlockedError(f"bearing overlap across splits: {details}")

    canonical_manifest = _canonical_records(manifest)
    file_hashes = {
        record["path"]: _sha256_file(record["path"])
        for record in canonical_manifest
    }
    payload = {
        "status": "passed",
        "protocol_mode": protocol_mode,
        "bearing_disjoint": bearing_disjoint,
        "split_semantics": (
            "bearing_disjoint"
            if bearing_disjoint
            else "condition_split_allows_bearing_reuse"
        ),
        "condition_split": _condition_split(normalized_splits),
        "record_count": len(canonical_manifest),
        "fault_class_counts": _fault_class_counts(canonical_manifest),
        "bearing_intersections": intersections,
        "file_hashes": dict(sorted(file_hashes.items())),
        "manifest_hash": _hash_payload(canonical_manifest),
        "split_hashes": {
            name: _hash_payload(records)
            for name, records in sorted(normalized_splits.items())
        },
        "splits": {
            name: records for name, records in sorted(normalized_splits.items())
        },
    }

    output_path = Path(path)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _normalized_text(value):
    return str(value).strip()


def _resolve_protocol_semantics(bearing_disjoint, protocol_mode):
    if bearing_disjoint is not None and not isinstance(bearing_disjoint, bool):
        raise ProtocolBlockedError("bearing_disjoint must be a boolean")
    if protocol_mode is not None:
        protocol_mode = _normalized_text(protocol_mode)
        if protocol_mode not in _PROTOCOL_MODES:
            raise ProtocolBlockedError(
                "protocol_mode must be 'bearing_disjoint' or 'paper_condition'"
            )

    if bearing_disjoint is None and protocol_mode is None:
        return True, DEFAULT_PROTOCOL_MODE
    if bearing_disjoint is None:
        return protocol_mode == DEFAULT_PROTOCOL_MODE, protocol_mode
    if protocol_mode is None:
        return bearing_disjoint, (
            DEFAULT_PROTOCOL_MODE
            if bearing_disjoint
            else PAPER_CONDITION_PROTOCOL_MODE
        )

    expected = protocol_mode == DEFAULT_PROTOCOL_MODE
    if bearing_disjoint != expected:
        raise ProtocolBlockedError(
            "bearing_disjoint and protocol_mode are inconsistent; "
            "condition-mode widening must be explicit"
        )
    return bearing_disjoint, protocol_mode


def _normalized_condition_ids(values, role):
    if isinstance(values, str):
        values = (values,)
    try:
        normalized = tuple(_normalized_text(value) for value in values)
    except TypeError as exc:
        raise ProtocolBlockedError(f"{role} condition IDs must be a non-empty iterable") from exc
    if not normalized or any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        raise ProtocolBlockedError(f"{role} condition IDs must be unique and non-empty")
    return normalized


def _normalized_column(manifest, column):
    return manifest[column].map(_normalized_text)


def _canonical_records(manifest):
    return sorted(
        (
            {
                column: (
                    _canonical_path(row[column])
                    if column == "path"
                    else _normalized_text(row[column])
                )
                for column in REQUIRED_MANIFEST_COLUMNS
            }
            for _, row in manifest.loc[:, REQUIRED_MANIFEST_COLUMNS].iterrows()
        ),
        key=lambda record: record["record_id"],
    )


def _canonical_path(value):
    source_path = Path(_normalized_text(value))
    if not source_path.is_absolute() or not source_path.is_file():
        raise ProtocolBlockedError("manifest paths must be absolute existing files")
    return str(source_path.resolve())


def _bearing_intersections(bearing_sets):
    return {
        f"{left}__{right}": sorted(bearing_sets[left].intersection(bearing_sets[right]))
        for left, right in combinations(sorted(bearing_sets), 2)
    }


def _condition_split(normalized_splits):
    return {
        name: sorted({record["condition_id"] for record in records})
        for name, records in sorted(normalized_splits.items())
    }


def _validate_split_allocation(manifest, normalized_splits):
    manifest_records = {
        record["record_id"]: record for record in _canonical_records(manifest)
    }
    split_ids = [
        record["record_id"]
        for records in normalized_splits.values()
        for record in records
    ]
    if len(split_ids) != len(set(split_ids)):
        raise ProtocolBlockedError("each record_id must appear in exactly one split")
    for records in normalized_splits.values():
        for record in records:
            if manifest_records.get(record["record_id"]) != record:
                raise ProtocolBlockedError(
                    f"split record {record['record_id']!r} does not match manifest"
                )
    if set(split_ids) != set(manifest_records):
        raise ProtocolBlockedError("split frames must allocate every manifest record exactly once")


def _fault_class_counts(records):
    counts = {}
    for record in records:
        label = record["fault_label"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _sha256_file(path):
    source_path = Path(path)
    if not source_path.is_file():
        raise ProtocolBlockedError(f"cannot hash missing source file: {source_path}")
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(value):
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
