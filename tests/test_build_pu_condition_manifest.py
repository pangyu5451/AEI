import json
import shutil

import pytest

from AGG_FWC.datasets.build_pu_condition_manifest import (
    AGG_PU_CLASS_NAMES,
    build_pu_condition_manifest,
)
from AGG_FWC.protocol import ProtocolBlockedError


SOURCE_CONDITIONS = (
    "N15_M01_F10",
    "N15_M07_F04",
    "N15_M07_F10",
)
TARGET_CONDITION = "N09_M07_F10"


def _make_pu_tree(tmp_path, *, include_all=True):
    root = tmp_path / "PU"
    bearings = AGG_PU_CLASS_NAMES if include_all else AGG_PU_CLASS_NAMES[:1]
    for bearing in bearings:
        class_dir = root / bearing / bearing
        class_dir.mkdir(parents=True)
        for condition in SOURCE_CONDITIONS:
            for ordinal in (2, 10):
                (class_dir / f"{condition}_{bearing}_{ordinal}.mat").write_bytes(
                    f"{condition}-{bearing}-{ordinal}".encode()
                )
        (class_dir / f"{TARGET_CONDITION}_{bearing}_1.mat").write_bytes(b"target")
    # This directory is present in the real archive but is not one of the
    # fourteen AGG classes and must not enter the candidate manifest.
    extra = root / "K001" / "K001"
    extra.mkdir(parents=True)
    (extra / "N15_M01_F10_K001_1.mat").write_bytes(b"extra")
    return root


def test_missing_data_root_is_blocked(tmp_path):
    with pytest.raises(ProtocolBlockedError, match="data_root.*does not exist"):
        build_pu_condition_manifest(
            data_root=tmp_path / "missing",
            manifest_path=tmp_path / "candidate.csv",
            audit_path=tmp_path / "audit.json",
        )


def test_condition_manifest_assigns_numeric_last_source_record_to_validation(tmp_path):
    root = _make_pu_tree(tmp_path)

    result = build_pu_condition_manifest(
        data_root=root,
        manifest_path=tmp_path / "candidate.csv",
        audit_path=tmp_path / "audit.json",
    )

    frame = result.manifest
    assert len(frame) == 98  # 14 * (3 source conditions * 2 + 1 target)
    assert set(frame["split"]) == {"source_train", "source_val", "target"}
    for condition in SOURCE_CONDITIONS:
        selected = frame[
            (frame["condition_id"] == condition)
            & (frame["bearing_id"] == AGG_PU_CLASS_NAMES[0])
        ]
        assert selected.loc[selected["split"] == "source_val", "record_ordinal"].tolist() == [10]
        assert selected.loc[selected["split"] == "source_train", "record_ordinal"].tolist() == [2]

    assert not frame["path"].str.contains("K001").any()
    assert frame["record_id"].is_unique
    assert set(frame["fault_label"]) == {
        f"bearing_class:{bearing}" for bearing in AGG_PU_CLASS_NAMES
    }
    assert frame["label_provenance"].str.contains("PU_bearing.py::cls_names").all()
    assert (frame["label_provenance"].str.contains("physical fault") ).all()


def test_manifest_generation_invokes_explicit_paper_condition_audit(tmp_path):
    root = _make_pu_tree(tmp_path)

    result = build_pu_condition_manifest(
        data_root=root,
        manifest_path=tmp_path / "candidate.csv",
        audit_path=tmp_path / "audit.json",
    )

    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert result.loader_bundle.protocol_mode == "paper_condition"
    assert result.loader_bundle.bearing_disjoint is False
    assert audit["status"] == "passed"
    assert audit["protocol_mode"] == "paper_condition"
    assert audit["bearing_disjoint"] is False
    assert audit["split_semantics"] == "condition_split_allows_bearing_reuse"
    assert audit["bearing_intersections"]["test__train"]
    assert audit["bearing_intersections"]["test__val"]


def test_invalid_mat_filename_is_blocked_in_scanned_class_directory(tmp_path):
    root = _make_pu_tree(tmp_path)
    bad_path = root / AGG_PU_CLASS_NAMES[0] / AGG_PU_CLASS_NAMES[0] / "bad.mat"
    bad_path.write_bytes(b"bad")

    with pytest.raises(ProtocolBlockedError, match="cannot parse.*bad.mat"):
        build_pu_condition_manifest(
            data_root=root,
            manifest_path=tmp_path / "candidate.csv",
            audit_path=tmp_path / "audit.json",
        )


def test_directory_links_keep_logical_record_ids_and_use_physical_audit_paths(tmp_path):
    root = _make_pu_tree(tmp_path)
    logical_class_dir = root / AGG_PU_CLASS_NAMES[0] / AGG_PU_CLASS_NAMES[0]
    physical_class_dir = tmp_path / "external" / AGG_PU_CLASS_NAMES[0]
    physical_class_dir.parent.mkdir(parents=True)
    shutil.move(str(logical_class_dir), str(physical_class_dir))
    try:
        logical_class_dir.symlink_to(physical_class_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    result = build_pu_condition_manifest(
        data_root=root,
        manifest_path=tmp_path / "linked-candidate.csv",
        audit_path=tmp_path / "linked-audit.json",
    )

    first = result.manifest[result.manifest["bearing_id"] == AGG_PU_CLASS_NAMES[0]].iloc[0]
    assert first["record_id"].startswith(f"{AGG_PU_CLASS_NAMES[0]}/{AGG_PU_CLASS_NAMES[0]}/")
    assert str(physical_class_dir) in first["path"]
