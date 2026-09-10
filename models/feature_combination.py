from dataclasses import dataclass, field, replace
import copy
import math
from numbers import Integral
from types import MappingProxyType

import torch
import torch.nn as nn

from AGG_FWC.models.fwc_types import Item


_LEGAL_SIZES = MappingProxyType({
    "gray": (1, 2),
    "green": (1, 2),
    "blue": (1, 2, 3, 4, 6),
    "purple": (1, 2, 4, 6),
    "gold": (1, 2, 3, 4, 6),
    "red": (1, 2, 4, 6, 9),
})
# Public for inspection; assignment always reads the private immutable canonical map.
LEGAL_SIZES = _LEGAL_SIZES

_COLOR_QUOTAS = (
    ("gray", 154),
    ("green", 128),
    ("blue", 102),
    ("purple", 67),
    ("gold", 41),
    ("red", 20),
)

_RED_QUALITY_THRESHOLD = 0.80
# This covers float32 representation and three-term weighted-sum roundoff only.
_RED_QUALITY_EPSILON = 1e-7
_COLLECTIBLE_COLORS = frozenset(("gray", "green", "blue", "purple", "gold"))


def assign_items(contribution, stability, redundancy):
    """Assign 512 feature scores to immutable, deterministic packing items."""
    _validate_scores(contribution, "contribution")
    _validate_scores(stability, "stability")
    _validate_scores(redundancy, "redundancy")
    if contribution.shape != stability.shape or contribution.shape != redundancy.shape:
        raise ValueError("contribution, stability, and redundancy must have the same shape")

    contribution_values = contribution.detach().cpu().tolist()
    stability_values = stability.detach().cpu().tolist()
    redundancy_values = redundancy.detach().cpu().tolist()
    records = []
    for c_values, s_values, r_values in zip(
        contribution_values, stability_values, redundancy_values
    ):
        records.append(_assign_record(c_values, s_values, r_values))
    return tuple(records)


def _validate_scores(scores, name):
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2 or scores.shape[1] != 512:
        raise ValueError(f"{name} must have shape [B, 512]")
    if not torch.is_floating_point(scores):
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(scores).all().item() or not torch.logical_and(scores >= 0.0, scores <= 1.0).all().item():
        raise ValueError(f"{name} values must be finite and within [0, 1]")


def _assign_record(c_values, s_values, r_values):
    utilities = [c * s * (1.0 - r) for c, s, r in zip(c_values, s_values, r_values)]
    ranked_indices = sorted(range(512), key=lambda index: (c_values[index], index))
    assigned_colors = {}
    cursor = 0
    for color, quota in _COLOR_QUOTAS:
        candidate_indices = ranked_indices[cursor : cursor + quota]
        cursor += quota
        if color == "red":
            for index in candidate_indices:
                quality = 0.5 * c_values[index] + 0.3 * s_values[index] + 0.2 * (1.0 - r_values[index])
                assigned_colors[index] = (
                    "red"
                    if quality >= _RED_QUALITY_THRESHOLD - _RED_QUALITY_EPSILON
                    else "gold"
                )
        else:
            assigned_colors.update((index, color) for index in candidate_indices)

    red_candidates = ranked_indices[-20:]
    red_candidate_median = _median([utilities[index] for index in red_candidates])
    sizes = {}
    for color, legal_sizes in _LEGAL_SIZES.items():
        color_indices = sorted(
            (index for index, assigned_color in assigned_colors.items() if assigned_color == color),
            key=lambda index: (r_values[index], index),
        )
        for rank, index in enumerate(color_indices):
            size_rank = rank * len(legal_sizes) // len(color_indices)
            if color == "red" and utilities[index] >= red_candidate_median:
                size_rank = len(legal_sizes) - 1 - size_rank
            sizes[index] = legal_sizes[size_rank]

    return tuple(
        Item(
            index=index,
            color=assigned_colors[index],
            size=sizes[index],
            utility=utilities[index],
            c=c_values[index],
            s=s_values[index],
            r=r_values[index],
        )
        for index in range(512)
    )


def _median(values):
    values = sorted(values)
    midpoint = len(values) // 2
    return (values[midpoint - 1] + values[midpoint]) / 2.0


class IdentityFeatureCombination(nn.Module):
    def forward(self, features):
        return features


@dataclass(frozen=True)
class FWCInferenceResult:
    """Deterministic record-level FWC output."""

    record_id: str
    window_gates: torch.Tensor
    items: tuple
    layout: object
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FWCBatchInferenceResult:
    """Window-order-preserving collection of independently gated records."""

    window_gates: torch.Tensor
    record_ids: tuple
    record_results: tuple


class FWCFeatureCombination(nn.Module):
    """Build one FWC layout and soft gate for every window in a record.

    ``forward`` treats its input batch as the windows of one default record
    and returns gated features.  Call ``record_gate`` when the record result
    and diagnostics are needed explicitly.
    """

    def __init__(self, attribute_predictor, external_indices=(), source_reports=None):
        super().__init__()
        self.attribute_predictor = attribute_predictor
        self.external_indices = _validate_external_indices(external_indices)
        if source_reports is None:
            self.source_reports = {}
        elif hasattr(source_reports, "c_quantiles"):
            self.source_reports = {"source": source_reports}
        elif hasattr(source_reports, "items"):
            self.source_reports = dict(source_reports)
        else:
            reports = tuple(source_reports)
            self.source_reports = {report.condition_id: report for report in reports}
        if any(index < 0 or index >= 512 for index in self.external_indices):
            raise ValueError("external feature indices must be in [0, 511]")

    def record_gate(self, record_features, record_id):
        features = _validate_record_features(record_features)
        contribution, stability, redundancy = self.attribute_predictor(features)
        scores = tuple(
            _validate_attribute(attribute, name, features.shape[0])
            for attribute, name in zip(
                (contribution, stability, redundancy),
                ("contribution", "stability", "redundancy"),
            )
        )
        averaged = tuple(attribute.mean(dim=0, keepdim=True) for attribute in scores)
        items = assign_items(*averaged)[0]
        from AGG_FWC.models.fwc_packing import exact_pack

        layout = exact_pack(item for item in items if item.index not in self.external_indices)
        return _result_with_gate(
            record_features=features,
            record_id=record_id,
            items=items,
            layout=layout,
            external_indices=self.external_indices,
        )

    def forward(self, record_features, record_id=None):
        result = self.record_gate(
            record_features,
            record_id="__default_record__" if record_id is None else record_id,
        )
        return record_features * result.window_gates

    def apply_collectibles(
        self,
        record_features,
        record_id,
        *,
        classifier,
        condition_id=None,
        mask_classifier=None,
        source_report=None,
    ):
        """Apply source-calibrated collectible selection and exactly one repack.

        ``classifier`` and ``mask_classifier`` receive floating-point feature
        batches with shape ``[B, 512]`` and return logits or probabilities.
        If no mask callback is supplied, the classifier callback is reused.
        """
        if not callable(classifier):
            raise ValueError("classifier must be callable")
        features = _validate_record_features(record_features)
        initial = self.record_gate(features, record_id)
        weighted_features = features * initial.window_gates
        initial_probabilities = _probabilities(
            classifier(weighted_features), features, expected_rows=features.shape[0]
        )
        num_classes = initial_probabilities.shape[1]
        preliminary_probabilities = initial_probabilities.mean(dim=0)
        preliminary_class = int(torch.argmax(preliminary_probabilities).item())

        report = source_report if source_report is not None else self._source_report(condition_id)
        c_quantiles, delta_quantiles = _report_quantiles(report)
        candidates = tuple(
            item
            for item in initial.items
            if item.color in _COLLECTIBLE_COLORS
            and item.index not in self.external_indices
        )
        mask_classifier = classifier if mask_classifier is None else mask_classifier
        if candidates:
            masked_features = weighted_features.unsqueeze(0).repeat(len(candidates), 1, 1)
            for candidate_position, item in enumerate(candidates):
                masked_features[candidate_position, :, item.index] = 0.0
            masked_outputs = mask_classifier(masked_features.reshape(-1, 512))
            masked_probabilities = _probabilities(
                masked_outputs,
                features,
                expected_rows=len(candidates) * features.shape[0],
                expected_columns=num_classes,
            ).reshape(len(candidates), features.shape[0], -1)
            initial_mean = initial_probabilities[:, preliminary_class].mean()
            masked_mean = masked_probabilities[:, :, preliminary_class].mean(dim=1)
            deltas = torch.clamp(initial_mean - masked_mean, min=0.0)
        else:
            deltas = torch.empty(0, dtype=features.dtype, device=features.device)

        decisions = []
        scored = []
        for item, delta in zip(candidates, deltas.detach().cpu().tolist()):
            delta_clipped = _quantile_clip(delta, delta_quantiles)
            c_clipped = _quantile_clip(item.c, c_quantiles)
            eligible = delta_clipped >= 0.40 and c_clipped >= 0.40
            key = 0.4 * delta_clipped + 0.6 * c_clipped if eligible else 0.0
            scored.append((key, item.index, item))
            decisions.append(
                {
                    "index": item.index,
                    "color": item.color,
                    "delta_margin": float(delta),
                    "delta_clipped": delta_clipped,
                    "c_hat": float(item.c),
                    "c_clipped": c_clipped,
                    "eligible": eligible,
                    "K": key,
                    "decision": "eligible" if eligible else "rejected",
                    "c_quantiles": list(c_quantiles),
                    "delta_margin_quantiles": list(delta_quantiles),
                }
            )

        eligible = sorted((entry for entry in scored if entry[0] >= 0.40), key=lambda value: (-value[0], value[1]))
        selected = []
        if eligible and eligible[0][0] >= 0.65:
            selected.append(eligible[0])
            if len(eligible) > 1 and eligible[1][0] >= 0.55:
                selected.append(eligible[1])
        selected_indices = {entry[1] for entry in selected}
        eligible_ranks = {entry[1]: rank for rank, entry in enumerate(eligible)}
        for decision in decisions:
            if decision["index"] in selected_indices:
                decision["decision"] = "selected"
                decision["elimination_reason"] = None
            elif not decision["eligible"]:
                decision["decision"] = "rejected"
                decision["elimination_reason"] = "dual_gate_failed"
            elif not eligible or eligible[0][0] < 0.65:
                decision["decision"] = "rejected"
                decision["elimination_reason"] = "top_K_below_0.65"
            elif eligible_ranks[decision["index"]] == 1 and decision["K"] < 0.55:
                decision["decision"] = "rejected"
                decision["elimination_reason"] = "second_K_below_0.55"
            else:
                decision["decision"] = "rejected"
                decision["elimination_reason"] = "only_top_two_collectibles"

        effective_items = tuple(
            replace(
                item,
                utility=item.utility
                * (1.0 + 9.0 * _clip((key - 0.55) / 0.45, 0.0, 1.0)),
            )
            if item.index in selected_indices
            else item
            for key, _, item in sorted(scored, key=lambda value: value[1])
        )
        unchanged_items = tuple(item for item in initial.items if item.index not in {entry[1] for entry in scored})
        effective_items = tuple(sorted(effective_items + unchanged_items, key=lambda item: item.index))
        from AGG_FWC.models.fwc_packing import exact_pack

        final_layout = exact_pack(
            item for item in effective_items if item.index not in self.external_indices
        )
        effective_by_index = {item.index: item for item in effective_items}
        selected_item_details = []
        for key, index, item in selected:
            multiplier = 1.0 + 9.0 * _clip((key - 0.55) / 0.45, 0.0, 1.0)
            selected_item_details.append(
                {
                    "index": index,
                    "raw_utility": float(item.utility),
                    "effective_utility": float(effective_by_index[index].utility),
                    "multiplier": multiplier,
                }
            )
        final_selected_item_details = []
        for item in effective_items:
            if item.index in set(final_layout.selected_indices).union(self.external_indices):
                raw_item = next(original for original in initial.items if original.index == item.index)
                multiplier = item.utility / raw_item.utility if raw_item.utility else 1.0
                final_selected_item_details.append(
                    {
                        "index": item.index,
                        "raw_utility": float(raw_item.utility),
                        "effective_utility": float(item.utility),
                        "multiplier": float(multiplier),
                    }
                )
        final_probabilities = _probabilities(
            classifier(
                features * _gate_tensor(features, effective_items, final_layout, self.external_indices)
            ),
            features,
            expected_rows=features.shape[0],
            expected_columns=num_classes,
        )
        final_record_probabilities = final_probabilities.mean(dim=0)
        diagnostics = {
            "initial_layout": _layout_to_json(initial.layout),
            "final_layout": _layout_to_json(final_layout),
            "external_features": sorted(self.external_indices),
            "red_priority": {
                "policy": "prefer large red within 95 percent of optimum",
                "threshold": 0.95,
                "applied": bool(final_layout.contains_large_red),
                "decision": "large_red_preferred" if final_layout.contains_large_red else "value_optimum",
                "initial_contains_large_red": bool(initial.layout.contains_large_red),
                "final_contains_large_red": bool(final_layout.contains_large_red),
            },
            "preliminary_class": preliminary_class,
            "initial_window_probabilities": initial_probabilities.detach().cpu().tolist(),
            "preliminary_probabilities": preliminary_probabilities.detach().cpu().tolist(),
            "initial_record_probabilities": preliminary_probabilities.detach().cpu().tolist(),
            "candidates": copy.deepcopy(decisions),
            "decisions": copy.deepcopy(decisions),
            "selected_items": sorted(selected_indices),
            "selected_collectibles": sorted(selected_indices),
            "selected_item_details": selected_item_details,
            "final_selected_item_details": final_selected_item_details,
            "final_window_probabilities": final_probabilities.detach().cpu().tolist(),
            "final_record_probabilities": final_record_probabilities.detach().cpu().tolist(),
            "final_record_prediction": int(torch.argmax(final_record_probabilities).item()),
            "repacked": True,
            "repack_marker": "one_repack",
            "repack_count": 1,
        }
        return _result_with_gate(
            record_features=features,
            record_id=record_id,
            items=effective_items,
            layout=final_layout,
            external_indices=self.external_indices,
            diagnostics=diagnostics,
        )

    def batch_record_gate(self, record_features, record_ids):
        """Gate mixed windows by grouping on explicit IDs and restoring order."""
        features = _validate_record_features(record_features)
        ids = tuple(record_ids)
        if not ids or len(ids) != features.shape[0]:
            raise ValueError("record_ids must contain one ID per feature window")
        if any(not isinstance(record_id, (str, int)) for record_id in ids):
            raise ValueError("record_ids must contain string or integer IDs")

        grouped_indices = {}
        for window_index, record_id in enumerate(ids):
            grouped_indices.setdefault(record_id, []).append(window_index)
        batch_gates = torch.empty_like(features)
        record_results = []
        for record_id, indices in grouped_indices.items():
            result = self.record_gate(features[indices], record_id=record_id)
            batch_gates[indices] = result.window_gates
            record_results.append(result)
        return FWCBatchInferenceResult(
            window_gates=batch_gates,
            record_ids=tuple(grouped_indices),
            record_results=tuple(record_results),
        )

    def _source_report(self, condition_id):
        if not self.source_reports:
            raise ValueError("source calibration report is required for collectibles")
        if condition_id in self.source_reports:
            return self.source_reports[condition_id]
        if condition_id is None and len(self.source_reports) == 1:
            return next(iter(self.source_reports.values()))
        raise ValueError(f"no source calibration report for condition {condition_id!r}")


def _validate_record_features(record_features):
    if not isinstance(record_features, torch.Tensor):
        raise ValueError("record_features must be a tensor with shape [W, 512]")
    if record_features.ndim != 2 or record_features.shape[1] != 512 or record_features.shape[0] == 0:
        raise ValueError("record_features must have shape [W, 512]")
    if not torch.is_floating_point(record_features) or not torch.isfinite(record_features).all().item():
        raise ValueError("record_features must be finite floating-point values")
    return record_features


def _validate_attribute(attribute, name, window_count):
    if not isinstance(attribute, torch.Tensor):
        raise ValueError(f"{name} must be a tensor with shape [W, 512]")
    if attribute.shape != (window_count, 512):
        raise ValueError(f"{name} must have shape [W, 512]")
    if not torch.is_floating_point(attribute):
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(attribute).all().item() or not torch.logical_and(attribute >= 0.0, attribute <= 1.0).all().item():
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return attribute


def _validate_external_indices(external_indices):
    try:
        values = tuple(external_indices)
    except TypeError as error:
        raise ValueError("external_indices must contain non-negative integers") from error
    validated = set()
    for index in values:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("external_indices must contain non-negative integer values")
        index = int(index)
        if index < 0 or index >= 512:
            raise ValueError("external feature indices must be in [0, 511]")
        validated.add(index)
    return frozenset(validated)


def _result_with_gate(record_features, record_id, items, layout, external_indices, diagnostics=None):
    gate_values = _gate_tensor(record_features, items, layout, external_indices)
    return FWCInferenceResult(
        record_id=str(record_id),
        window_gates=gate_values,
        items=tuple(items),
        layout=layout,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _gate_tensor(record_features, items, layout, external_indices):
    selected = set(layout.selected_indices).union(external_indices)
    gate_values = torch.ones(512, dtype=record_features.dtype, device=record_features.device)
    for item in items:
        if item.index in selected:
            gate_values[item.index] = 1.0 + math.log1p(item.size * item.utility)
    return gate_values.unsqueeze(0).repeat(record_features.shape[0], 1)


def _layout_to_json(layout):
    return {
        "selected_indices": [int(index) for index in layout.selected_indices],
        "placements": [
            {
                "item_index": int(placement.item_index),
                "mask": int(placement.mask),
                "shape": [int(value) for value in placement.shape],
            }
            for placement in layout.placements
        ],
        "total_utility": float(layout.total_utility),
        "cells_used": int(layout.cells_used),
        "contains_large_red": bool(layout.contains_large_red),
    }


def _report_quantiles(report):
    if isinstance(report, dict):
        c_quantiles = report.get("c_quantiles")
        delta_quantiles = report.get("delta_margin_quantiles")
    else:
        c_quantiles = getattr(report, "c_quantiles", None)
        delta_quantiles = getattr(report, "delta_margin_quantiles", None)
    if c_quantiles is None or delta_quantiles is None:
        raise ValueError("source calibration must provide C and delta-margin quantiles")
    return _valid_quantiles(c_quantiles, "c_quantiles"), _valid_quantiles(
        delta_quantiles, "delta_margin_quantiles"
    )


def _valid_quantiles(values, name):
    values = tuple(float(value) for value in values)
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain two finite values")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} values must be within [0, 1]")
    if values[1] < values[0]:
        raise ValueError(f"{name} must be ordered")
    return values


def _quantile_clip(value, quantiles):
    lower, upper = quantiles
    if upper == lower:
        return 1.0 if value >= upper else 0.0
    return _clip((float(value) - lower) / (upper - lower), 0.0, 1.0)


def _clip(value, lower, upper):
    return min(max(float(value), lower), upper)


def _probabilities(output, reference, *, expected_rows, expected_columns=None):
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise ValueError("classifier must return a two-dimensional tensor")
    if output.shape[0] != expected_rows:
        raise ValueError(
            f"classifier output must have {expected_rows} rows, got {output.shape[0]}"
        )
    if expected_columns is not None and output.shape[1] != expected_columns:
        raise ValueError(
            f"classifier output must use the same classes; expected {expected_columns}, got {output.shape[1]}"
        )
    if output.shape[1] == 0:
        raise ValueError("classifier output must contain at least one class")
    output = output.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(output).all().item():
        raise ValueError("classifier output must be finite")
    if torch.all(output >= 0.0).item() and torch.allclose(
        output.sum(dim=1), torch.ones(output.shape[0], device=output.device, dtype=output.dtype), atol=1e-5
    ):
        return output
    return torch.softmax(output, dim=1)
