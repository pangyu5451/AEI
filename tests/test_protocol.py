import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from AGG_FWC.config import build_config, prepare_artifact_dirs
from AGG_FWC.protocol import (
    ProtocolBlockedError,
    assert_disjoint_bearings,
    validate_manifest,
    write_protocol_audit,
)


def _manifest(tmp_path):
    rows = []
    for record_id, condition_id, bearing_id, fault_label, contents in (
        ("r1", "N15_M01_F10", "KA01", "normal", b"first"),
        ("r2", "N15_M01_F10", "KA02", "inner_race", b"second"),
        ("r3", "N15_M07_F04", "KA03", "outer_race", b"third"),
        ("r4", "N09_M07_F10", "KA04", "normal", b"fourth"),
    ):
        source_path = tmp_path / f"{record_id}.mat"
        source_path.write_bytes(contents)
        rows.append(
            {
                "record_id": record_id,
                "condition_id": condition_id,
                "bearing_id": bearing_id,
                "fault_label": fault_label,
                "label_provenance": f"PU profile reference for {bearing_id}",
                "path": str(source_path),
            }
        )
    return pd.DataFrame(rows)


def test_valid_manifest_split_passes_and_writes_deterministic_audit(tmp_path):
    manifest = _manifest(tmp_path)
    split_frames = {
        "train": manifest.iloc[:2].copy(),
        "val": manifest.iloc[2:3].copy(),
        "test": manifest.iloc[3:].copy(),
    }

    validate_manifest(manifest)
    assert_disjoint_bearings(**split_frames)
    first_path = tmp_path / "first-audit.json"
    second_path = tmp_path / "second-audit.json"
    write_protocol_audit(manifest, split_frames, first_path)
    write_protocol_audit(manifest, split_frames, second_path)

    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["record_count"] == 4
    assert payload["fault_class_counts"] == {
        "inner_race": 1,
        "normal": 2,
        "outer_race": 1,
    }
    assert payload["bearing_intersections"] == {
        "test__train": [],
        "test__val": [],
        "train__val": [],
    }
    assert set(payload["file_hashes"]) == set(manifest["path"])
    assert set(payload["split_hashes"]) == {"train", "val", "test"}
    assert len(payload["manifest_hash"]) == 64
    assert first_path.read_bytes() == second_path.read_bytes()


def test_protocol_audit_generator_output_is_accepted_by_strict_runner(tmp_path):
    from AGG_FWC.train import StrictStagedRunner

    manifest = _manifest(tmp_path)
    split_frames = {
        "train": manifest.iloc[:2].copy(),
        "val": manifest.iloc[2:3].copy(),
        "test": manifest.iloc[3:].copy(),
    }
    audit_path = tmp_path / "audit.json"
    write_protocol_audit(manifest, split_frames, audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    config = build_config(output_root=tmp_path, run_id="protocol-strict-chain", protocol_mode="strict")
    prepare_artifact_dirs(config)
    component_path = tmp_path / "component.bin"
    component_path.write_bytes(b"component")
    component_hash = hashlib.sha256(component_path.read_bytes()).hexdigest()

    class Loader:
        def __init__(self, partition, record_ids):
            self.partition = partition
            self.record_level = True
            self.record_ids = record_ids

        def __iter__(self):
            return iter(())

    runner = StrictStagedRunner(
        config,
        source_train_loader=Loader("source", ["r1", "r2"]),
        source_val_loader=Loader("source", ["r3"]),
        target_loader=Loader("target", ["r4"]),
        num_classes=2,
        source_split={"train": ["r1", "r2"], "val": ["r3"]},
        target_split=["r4"],
        protocol_audit=audit,
        component_hashes={"model": component_hash},
        component_files={"model": component_path},
        protocol_mode="strict",
    )

    assert runner.protocol_audit["status"] == "passed"


def test_audit_contains_sorted_canonical_split_records(tmp_path):
    manifest = _manifest(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    uncanonical_path = str(nested / ".." / "r1.mat")
    manifest.loc[0, "path"] = uncanonical_path
    split_frames = {
        "train": manifest.iloc[:2].iloc[::-1].copy(),
        "val": manifest.iloc[2:3].copy(),
        "test": manifest.iloc[3:].copy(),
    }
    audit_path = tmp_path / "audit.json"

    write_protocol_audit(manifest, split_frames, audit_path)

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    canonical_path = str((tmp_path / "r1.mat").resolve())
    train_records = payload["splits"]["train"]
    assert [record["record_id"] for record in train_records] == ["r1", "r2"]
    assert train_records[0]["path"] == canonical_path
    assert train_records[0]["label_provenance"] == "PU profile reference for KA01"
    assert canonical_path in payload["file_hashes"]
    assert uncanonical_path not in payload["file_hashes"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda manifest: manifest.__setitem__(
                "record_id", ["r1", "r1", "r3", "r4"]
            ),
            "record_id",
        ),
        (lambda manifest: manifest.drop(columns="path", inplace=True), "missing required columns"),
        (
            lambda manifest: manifest.__setitem__(
                "fault_label", ["KA01", "inner_race", "outer_race", "normal"]
            ),
            "distinct from bearing_id",
        ),
    ],
)
def test_manifest_blocks_invalid_metadata(tmp_path, mutate, message):
    manifest = _manifest(tmp_path)
    mutate(manifest)

    with pytest.raises(ProtocolBlockedError, match=message):
        validate_manifest(manifest)


def test_assert_disjoint_bearings_blocks_overlap(tmp_path):
    manifest = _manifest(tmp_path)

    with pytest.raises(ProtocolBlockedError, match="bearing overlap"):
        assert_disjoint_bearings(
            train=manifest.iloc[:2].copy(),
            val=manifest.iloc[2:3].copy(),
            test=manifest.iloc[1:2].copy(),
        )


def test_audit_blocks_split_record_metadata_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    split_frames = {
        "train": manifest.iloc[:2].copy(),
        "val": manifest.iloc[2:3].copy(),
        "test": manifest.iloc[3:].copy(),
    }
    split_frames["val"].loc[:, "bearing_id"] = "KA99"

    with pytest.raises(ProtocolBlockedError, match="does not match manifest"):
        write_protocol_audit(manifest, split_frames, tmp_path / "audit.json")


def test_manifest_blocks_duplicate_canonical_source_paths(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[2, "path"] = manifest.loc[0, "path"]
    split_frames = {
        "train": manifest.iloc[:2].copy(),
        "val": manifest.iloc[2:3].copy(),
        "test": manifest.iloc[3:].copy(),
    }

    with pytest.raises(ProtocolBlockedError, match="path values must be unique"):
        write_protocol_audit(manifest, split_frames, tmp_path / "audit.json")


def test_manifest_rejects_relative_source_path(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[0, "path"] = "relative.mat"

    with pytest.raises(ProtocolBlockedError, match="absolute existing files"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda manifest: manifest.drop(columns="label_provenance", inplace=True),
            "label_provenance",
        ),
        (lambda manifest: manifest.__setitem__("label_provenance", "   "), "blank values"),
    ],
)
def test_manifest_requires_nonblank_label_provenance(tmp_path, mutate, message):
    manifest = _manifest(tmp_path)
    mutate(manifest)

    with pytest.raises(ProtocolBlockedError, match=message):
        validate_manifest(manifest)


def test_audit_requires_validation_split(tmp_path):
    manifest = _manifest(tmp_path)
    split_frames = {
        "train": manifest.iloc[:2].copy(),
        "test": manifest.iloc[2:].copy(),
    }

    with pytest.raises(ProtocolBlockedError, match="train, val, and test"):
        write_protocol_audit(manifest, split_frames, tmp_path / "audit.json")


def test_audit_rejects_unexpected_split_name(tmp_path):
    manifest = _manifest(tmp_path)
    split_frames = {
        "train": manifest.iloc[:1].copy(),
        "val": manifest.iloc[1:2].copy(),
        "test": manifest.iloc[2:3].copy(),
        "calibration": manifest.iloc[3:].copy(),
    }

    with pytest.raises(ProtocolBlockedError, match="train, val, and test"):
        write_protocol_audit(manifest, split_frames, tmp_path / "audit.json")


def test_paderborn_bearing_codes_cannot_be_used_as_fault_labels(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["fault_label"] = manifest["bearing_id"]

    with pytest.raises(ProtocolBlockedError, match="file-name label inference is blocked"):
        validate_manifest(manifest)
