"""Build the PU condition-transfer manifest used by the original AGG setup.

This script deliberately separates two notions that are often conflated in
fault-diagnosis code:

* the original AGG labels are the fourteen bearing classes listed in
  ``datasets/PU_bearing.py``; they are represented here as
  ``bearing_class:<bearing_id>`` and are **not** physical fault-type labels;
* the split is by operating condition, with the final numbered file for each
  source-condition/bearing pair held out as ``source_val``.

The resulting protocol is therefore ``paper_condition`` plus
``bearing_disjoint=False``.  The audit records cross-split bearing
intersections instead of presenting this as unseen-bearing generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd

from AGG_FWC.datasets.strict_pu_adapter import StrictPULoaderBundle, build_strict_pu_loaders
from AGG_FWC.protocol import ProtocolBlockedError


AGG_PU_CLASS_NAMES = (
    "KA04",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KI04",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
    "KB27",
    "KB23",
    "KB24",
)
AGG_PU_CLASS_INDEX = {
    bearing: index for index, bearing in enumerate(AGG_PU_CLASS_NAMES)
}
AGG_LABEL_PROVENANCE = (
    "AGG_FWC/datasets/PU_bearing.py::cls_names (explicit 14-class mapping); "
    "fault_label=bearing_class:<bearing_id>, not a physical fault type."
)
DEFAULT_SOURCE_CONDITIONS = (
    "N15_M01_F10",
    "N15_M07_F04",
    "N15_M07_F10",
)
DEFAULT_TARGET_CONDITION = "N09_M07_F10"
ALLOWED_CONDITIONS = frozenset((*DEFAULT_SOURCE_CONDITIONS, DEFAULT_TARGET_CONDITION))

_MAT_NAME = re.compile(
    r"^(?P<condition>N\d+_M\d+_F\d+)_(?P<bearing>(?:KA|KI|KB)\d+)_(?P<ordinal>\d+)\.mat$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PUConditionManifestBuild:
    """Candidate manifest, audit and metadata bundle produced by the builder."""

    manifest: pd.DataFrame
    manifest_path: Path
    audit_path: Path
    loader_bundle: StrictPULoaderBundle | None
    scanned_file_count: int

    @property
    def counts(self) -> dict[str, int]:
        return {
            "manifest_records": int(len(self.manifest)),
            "source_train_records": int((self.manifest["split"] == "source_train").sum()),
            "source_val_records": int((self.manifest["split"] == "source_val").sum()),
            "target_records": int((self.manifest["split"] == "target").sum()),
            "scanned_mat_files": self.scanned_file_count,
        }


def build_pu_condition_manifest(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    source_conditions: Sequence[str] = DEFAULT_SOURCE_CONDITIONS,
    target_condition: str = DEFAULT_TARGET_CONDITION,
    build_loader_bundle: bool = True,
) -> PUConditionManifestBuild:
    """Scan the four paper conditions and create an audited candidate manifest.

    Every ``.mat`` under one of the fourteen explicit AGG class directories is
    parsed from its real filename.  ``record_id`` is the POSIX relative path,
    so it remains unique without inventing an index.  The largest numeric
    suffix in each source condition/bearing group becomes ``source_val``;
    every other file in that group becomes ``source_train``.  All target files
    remain in ``target``.
    """

    root = _require_directory(data_root)
    source_conditions = _normalize_conditions(source_conditions, target_condition)
    target_condition = _normalize_condition(target_condition, "target_condition")
    if target_condition in source_conditions:
        raise ProtocolBlockedError("source conditions and target condition must be disjoint")

    rows = []
    scanned_file_count = 0
    for class_index, expected_bearing in enumerate(AGG_PU_CLASS_NAMES):
        class_dir = root / expected_bearing / expected_bearing
        if not class_dir.is_dir():
            raise ProtocolBlockedError(
                f"missing required AGG PU class directory: {class_dir}"
            )
        for path in sorted(class_dir.glob("*.mat"), key=lambda item: item.name.lower()):
            scanned_file_count += 1
            parsed = _parse_mat_name(path, expected_bearing)
            condition = parsed["condition"]
            if condition not in (*source_conditions, target_condition):
                continue
            # Keep the lexical path below the user-facing data_root for the
            # stable record identity.  Some PU archives expose class folders
            # as directory links; resolving before relative_to(root) would
            # incorrectly reject those valid logical paths.
            relative_path = path.absolute().relative_to(root.absolute()).as_posix()
            rows.append(
                {
                    "record_id": relative_path,
                    "condition_id": condition,
                    "bearing_id": expected_bearing,
                    "fault_label": f"bearing_class:{expected_bearing}",
                    "label_provenance": AGG_LABEL_PROVENANCE,
                    # Keep the logical path under data_root.  The archive can
                    # expose class folders through directory links; the
                    # protocol layer resolves this path only when hashing the
                    # real source file.
                    "path": str(path.absolute()),
                    "record_ordinal": parsed["ordinal"],
                    "class_index": class_index,
                    "file_name": path.name,
                }
            )

    if not rows:
        raise ProtocolBlockedError("no selected PU condition .mat files were found")
    manifest = _assign_splits(pd.DataFrame(rows), source_conditions, target_condition)
    if set(manifest["split"]) != {"source_train", "source_val", "target"}:
        raise ProtocolBlockedError(
            "selected PU files cannot produce non-empty source_train/source_val/target splits"
        )

    manifest_file = Path(manifest_path).expanduser().resolve()
    audit_file = Path(audit_path).expanduser().resolve()
    _write_new_manifest(manifest, manifest_file)
    loader_bundle = None
    if build_loader_bundle:
        loader_bundle = build_strict_pu_loaders(
            data_root=root,
            manifest_path=manifest_file,
            audit_path=audit_file,
            protocol_mode="paper_condition",
            bearing_disjoint=False,
        )
    return PUConditionManifestBuild(
        manifest=manifest,
        manifest_path=manifest_file,
        audit_path=audit_file,
        loader_bundle=loader_bundle,
        scanned_file_count=scanned_file_count,
    )


def _assign_splits(
    manifest: pd.DataFrame,
    source_conditions: Sequence[str],
    target_condition: str,
) -> pd.DataFrame:
    result = manifest.copy()
    result["split"] = "target"
    result.loc[result["condition_id"].isin(source_conditions), "split"] = "source_train"
    source_mask = result["condition_id"].isin(source_conditions)
    for (_, _), group in result.loc[source_mask].groupby(
        ["condition_id", "bearing_id"], sort=False
    ):
        ordered = group.sort_values(
            ["record_ordinal", "record_id"], kind="stable"
        )
        last_index = ordered.index[-1]
        result.loc[last_index, "split"] = "source_val"
    if (result["condition_id"] == target_condition).sum() == 0:
        raise ProtocolBlockedError(f"target condition has no .mat files: {target_condition}")
    if (result["split"] == "source_train").sum() == 0:
        raise ProtocolBlockedError("source conditions have no records left for source_train")
    return result.sort_values(
        ["condition_id", "bearing_id", "record_ordinal", "record_id"],
        key=lambda column: column.map(
            {condition: index for index, condition in enumerate((*source_conditions, target_condition))}
        )
        if column.name == "condition_id"
        else column,
        kind="stable",
    ).reset_index(drop=True)


def _parse_mat_name(path: Path, expected_bearing: str) -> dict[str, object]:
    match = _MAT_NAME.fullmatch(path.name)
    if match is None:
        raise ProtocolBlockedError(
            f"cannot parse PU condition/bearing/ordinal from .mat filename: {path.name}"
        )
    condition = match.group("condition").upper()
    bearing = match.group("bearing").upper()
    if bearing != expected_bearing:
        raise ProtocolBlockedError(
            f"PU filename bearing {bearing} disagrees with class directory {expected_bearing}"
        )
    return {"condition": condition, "bearing": bearing, "ordinal": int(match.group("ordinal"))}


def _normalize_conditions(values: Iterable[str], target_condition: str) -> tuple[str, ...]:
    try:
        normalized = tuple(_normalize_condition(value, "source_conditions") for value in values)
    except TypeError as exc:
        raise ProtocolBlockedError("source_conditions must be an iterable of condition IDs") from exc
    if len(normalized) != 3 or len(set(normalized)) != len(normalized):
        raise ProtocolBlockedError("source_conditions must contain three unique condition IDs")
    target = _normalize_condition(target_condition, "target_condition")
    if any(condition not in ALLOWED_CONDITIONS for condition in (*normalized, target)):
        raise ProtocolBlockedError(
            "PU condition IDs must be among N09_M07_F10, N15_M01_F10, "
            "N15_M07_F04, N15_M07_F10"
        )
    return normalized


def _normalize_condition(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolBlockedError(f"{name} must contain non-blank strings")
    return value.strip().upper().rstrip("_")


def _require_directory(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ProtocolBlockedError(f"data_root does not exist or is not a directory: {root}")
    return root


def _write_new_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate manifest: {path}")
    columns = [
        "record_id",
        "condition_id",
        "bearing_id",
        "fault_label",
        "label_provenance",
        "path",
        "split",
        "record_ordinal",
        "class_index",
        "file_name",
    ]
    manifest.loc[:, columns].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the PU paper condition-transfer manifest")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--source-conditions", default=",".join(DEFAULT_SOURCE_CONDITIONS))
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_pu_condition_manifest(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        audit_path=args.audit_path,
        source_conditions=tuple(item for item in args.source_conditions.split(",") if item.strip()),
        target_condition=args.target_condition,
    )
    print(json.dumps(result.counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
