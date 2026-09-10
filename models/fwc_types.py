"""Immutable data records used by the FWC calibration and packing stages."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Item:
    index: int
    color: str
    size: int
    utility: float
    c: float = 0.0
    s: float = 0.0
    r: float = 0.0


@dataclass(frozen=True)
class Placement:
    item_index: int
    mask: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class Layout:
    selected_indices: tuple[int, ...]
    placements: tuple[Placement, ...]
    total_utility: float
    cells_used: int
    contains_large_red: bool


@dataclass(frozen=True)
class ConditionAttributeReport:
    condition_id: str
    contribution: tuple[float, ...]
    stability: tuple[float, ...]
    redundancy: tuple[float, ...]
    c_quantiles: tuple[float, float]
    delta_margin_quantiles: tuple[float, float]

    def __post_init__(self):
        for field_name in ("contribution", "stability", "redundancy"):
            values = tuple(float(value) for value in getattr(self, field_name))
            if not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"{field_name} values must be finite and within [0, 1]")
            object.__setattr__(self, field_name, values)
        for field_name in ("c_quantiles", "delta_margin_quantiles"):
            values = tuple(float(value) for value in getattr(self, field_name))
            if len(values) != 2 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{field_name} must contain two finite values")
            if field_name == "c_quantiles" and any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("c_quantiles values must be within [0, 1]")
            object.__setattr__(
                self,
                field_name,
                values,
            )


@dataclass(frozen=True, init=False)
class AuditedSourcePartition:
    """Source/target partition from an audited manifest; not a security capability.

    This type prevents accidental mixing in a trusted runner.  Hostile code in
    the same Python interpreter can bypass it; Task7 owns loader enforcement.
    """

    source_record_ids: tuple[str, ...]
    source_condition_ids: tuple[str, ...]
    target_record_ids: tuple[str, ...]
    target_condition_ids: tuple[str, ...]


@dataclass(frozen=True, init=False)
class AuditedSourcePayload:
    """Fingerprint-bound source payload attestation for trusted-runner leak checks.

    It is intentionally not presented as unforgeable or as a defense against
    malicious in-process callers.
    """

    fingerprint: str


def _create_audited_source_partition(
    source_record_ids, source_condition_ids, target_record_ids, target_condition_ids
):
    partition = object.__new__(AuditedSourcePartition)
    for name, values in (
        ("source_record_ids", source_record_ids),
        ("source_condition_ids", source_condition_ids),
        ("target_record_ids", target_record_ids),
        ("target_condition_ids", target_condition_ids),
    ):
        object.__setattr__(partition, name, tuple(values))
    return partition


def _require_audited_source_partition(partition):
    if not isinstance(partition, AuditedSourcePartition):
        raise ValueError("an audited source partition is required")


def _create_audited_source_payload(fingerprint):
    payload = object.__new__(AuditedSourcePayload)
    object.__setattr__(payload, "fingerprint", str(fingerprint))
    return payload


def _verify_audited_source_payload(payload, fingerprint):
    _require_audited_source_payload(payload)
    if payload.fingerprint != fingerprint:
        raise ValueError("audited source attestation does not match these inputs")


def _require_audited_source_payload(payload):
    if not isinstance(payload, AuditedSourcePayload):
        raise ValueError("an audited source payload attestation is required")
