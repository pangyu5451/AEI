import json

import pandas as pd
import pytest

from AGG_FWC.protocol import ProtocolBlockedError
from AGG_FWC.datasets.strict_pu_adapter import (
    build_strict_pu_loaders,
    load_legacy_pu_windows,
)


def _manifest(tmp_path):
    rows = []
    for record_id, split, condition, bearing, label in (
        ("r-train", "source_train", "N15_M01_F10", "KA01", "normal"),
        ("r-val", "source_val", "N15_M07_F04", "KA02", "inner_race"),
        ("r-test", "target", "N09_M07_F10", "KA03", "outer_race"),
    ):
        path = tmp_path / f"{record_id}.mat"
        path.write_bytes(record_id.encode("ascii"))
        rows.append(
            {
                "record_id": record_id,
                "condition_id": condition,
                "bearing_id": bearing,
                "fault_label": label,
                "label_provenance": f"verified metadata for {bearing}",
                "path": str(path),
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def _overlapping_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[2, "bearing_id"] = manifest.loc[0, "bearing_id"]
    return manifest


def test_missing_manifest_path_is_blocked(tmp_path):
    with pytest.raises(ProtocolBlockedError, match="manifest.*does not exist"):
        build_strict_pu_loaders(
            data_root=tmp_path,
            manifest_path=tmp_path / "missing.csv",
            audit_path=tmp_path / "audit.json",
        )


def test_missing_data_root_is_blocked(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _manifest(tmp_path).to_csv(manifest_path, index=False)

    with pytest.raises(ProtocolBlockedError, match="data_root.*does not exist"):
        build_strict_pu_loaders(
            data_root=tmp_path / "missing-root",
            manifest_path=manifest_path,
            audit_path=tmp_path / "audit.json",
        )


def test_current_window_loader_cannot_be_promoted_without_bearing_metadata(tmp_path):
    with pytest.raises(ProtocolBlockedError, match="bearing-level|record_id"):
        load_legacy_pu_windows(
            data_root=tmp_path,
            condition_index=0,
            num_samples=2,
            bearing_metadata=None,
        )


def test_valid_manifest_returns_auditable_record_level_loader_metadata(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _manifest(tmp_path).to_csv(manifest_path, index=False)
    audit_path = tmp_path / "protocol_audit.json"

    bundle = build_strict_pu_loaders(
        data_root=tmp_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )

    assert bundle.source_train.partition == "source"
    assert bundle.source_train.record_level is True
    assert bundle.source_train.record_ids == ("r-train",)
    assert bundle.source_val.partition == "source"
    assert bundle.source_val.record_ids == ("r-val",)
    assert bundle.target.partition == "target"
    assert bundle.target.record_ids == ("r-test",)
    assert bundle.loader_metadata.keys() == {
        "source_train",
        "source_val",
        "target",
    }
    assert bundle.protocol_audit["status"] == "passed"
    assert json.loads(audit_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_manifest_without_split_or_bearing_metadata_is_blocked(tmp_path):
    manifest = _manifest(tmp_path).drop(columns=["split", "bearing_id"])
    manifest_path = tmp_path / "incomplete.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ProtocolBlockedError, match="split|bearing_id"):
        build_strict_pu_loaders(
            data_root=tmp_path,
            manifest_path=manifest_path,
            audit_path=tmp_path / "audit.json",
        )


def test_default_protocol_still_blocks_bearing_overlap(tmp_path):
    manifest_path = tmp_path / "overlap.csv"
    _overlapping_manifest(tmp_path).to_csv(manifest_path, index=False)

    with pytest.raises(ProtocolBlockedError, match="bearing overlap"):
        build_strict_pu_loaders(
            data_root=tmp_path,
            manifest_path=manifest_path,
            audit_path=tmp_path / "audit.json",
        )


def test_explicit_paper_condition_mode_accepts_overlap_and_audits_semantics(tmp_path):
    manifest_path = tmp_path / "overlap.csv"
    _overlapping_manifest(tmp_path).to_csv(manifest_path, index=False)
    audit_path = tmp_path / "paper-condition-audit.json"

    bundle = build_strict_pu_loaders(
        data_root=tmp_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
        protocol_mode="paper_condition",
    )

    assert bundle.protocol_audit["protocol_mode"] == "paper_condition"
    assert bundle.protocol_audit["bearing_disjoint"] is False
    assert bundle.protocol_mode == "paper_condition"
    assert bundle.bearing_disjoint is False
    assert bundle.protocol_audit["split_semantics"] == "condition_split_allows_bearing_reuse"
    assert bundle.protocol_audit["condition_split"] == {
        "test": ["N09_M07_F10"],
        "train": ["N15_M01_F10"],
        "val": ["N15_M07_F04"],
    }
    assert bundle.protocol_audit["bearing_intersections"]["test__train"] == ["KA01"]


def test_explicit_false_flag_accepts_paper_condition_overlap(tmp_path):
    manifest_path = tmp_path / "overlap.csv"
    _overlapping_manifest(tmp_path).to_csv(manifest_path, index=False)

    bundle = build_strict_pu_loaders(
        data_root=tmp_path,
        manifest_path=manifest_path,
        audit_path=tmp_path / "false-flag-audit.json",
        bearing_disjoint=False,
    )

    assert bundle.protocol_audit["protocol_mode"] == "paper_condition"
    assert bundle.protocol_audit["bearing_disjoint"] is False


def test_contradictory_protocol_parameters_are_blocked(tmp_path):
    manifest_path = tmp_path / "overlap.csv"
    _overlapping_manifest(tmp_path).to_csv(manifest_path, index=False)

    with pytest.raises(ProtocolBlockedError, match="inconsistent"):
        build_strict_pu_loaders(
            data_root=tmp_path,
            manifest_path=manifest_path,
            audit_path=tmp_path / "contradictory-audit.json",
            bearing_disjoint=True,
            protocol_mode="paper_condition",
        )
