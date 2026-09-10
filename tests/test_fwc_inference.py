import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from AGG_FWC.models import feature_combination
from AGG_FWC.models.agg_fwc_model import AGGFWCModel
from AGG_FWC.models.fwc_types import ConditionAttributeReport
from AGG_FWC.models.feature_combination import assign_items
from AGG_FWC.models.classifier import Classifier
from AGG_FWC.models.extractor import Feature_extractor


class _ConstantAttributePredictor:
    def __call__(self, features):
        scores = torch.full((features.shape[0], 512), 0.8, dtype=features.dtype)
        return scores, torch.ones_like(scores), torch.zeros_like(scores)


class _HighFirstAttributePredictor:
    def __call__(self, features):
        contribution = torch.zeros((features.shape[0], 512), dtype=features.dtype)
        contribution[:, 0] = 1.0
        return contribution, torch.ones_like(contribution), torch.zeros_like(contribution)


class _IntegerAttributePredictor:
    def __call__(self, features):
        integer_scores = torch.ones((features.shape[0], 512), dtype=torch.int64)
        return integer_scores, integer_scores, integer_scores


def _report(c_quantiles=(0.0, 1.0), delta_quantiles=(0.0, 1.0)):
    return ConditionAttributeReport(
        condition_id="source",
        contribution=tuple([0.8] * 512),
        stability=tuple([1.0] * 512),
        redundancy=tuple([0.0] * 512),
        c_quantiles=c_quantiles,
        delta_margin_quantiles=delta_quantiles,
    )


def _small_module(source_reports=None):
    keep = {1, 2, 492, 493}
    external = set(range(512)) - keep
    return feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(),
        external_indices=external,
        source_reports=source_reports,
    )


def _record(*active_indices):
    features = torch.zeros(3, 512)
    for index in active_indices:
        features[:, index] = 1.0
    return features


def _one_collectible_classifier(features):
    probability = torch.where(features[:, 1] > 0, torch.tensor(0.95), torch.tensor(0.5))
    return torch.stack((probability, 1.0 - probability), dim=1)


def _two_collectible_classifier(features):
    both_active = (features[:, 1] > 0) & (features[:, 2] > 0)
    probability = torch.where(both_active, torch.tensor(0.98), torch.tensor(0.5))
    return torch.stack((probability, 1.0 - probability), dim=1)


def _mixed_sign_margin_classifier(features):
    positive_candidate = features[:, 1] > 0
    negative_candidate = features[:, 1] < 0
    positive_baseline = features[:, 4] > 0
    probability = torch.where(
        positive_candidate,
        torch.tensor(0.9),
        torch.where(negative_candidate, torch.tensor(0.1), torch.where(positive_baseline, torch.tensor(0.9), torch.tensor(0.1))),
    )
    return torch.stack((probability, 1.0 - probability), dim=1)


class _FixedExtractor(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros(inputs.shape[0], 512)


def test_same_record_uses_one_layout_for_all_windows():
    assert hasattr(feature_combination, "FWCFeatureCombination")
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(), external_indices=range(1, 512)
    )

    result = module.record_gate(torch.randn(4, 512), record_id="r1")

    assert result.window_gates.shape == (4, 512)
    assert torch.equal(result.window_gates[0], result.window_gates[3])
    assert torch.all(result.window_gates >= 1)


def test_selected_feature_gate_is_not_identity():
    module = feature_combination.FWCFeatureCombination(
        _HighFirstAttributePredictor(), external_indices=range(1, 512)
    )

    result = module.record_gate(torch.ones(2, 512), record_id="r1")

    assert result.window_gates[0, 0] > 1.0
    assert torch.all(result.window_gates[:, 1:] >= 1.0)


def test_model_exposes_raw_features_and_optional_gate_without_changing_classifier():
    torch.manual_seed(2026)
    model = AGGFWCModel(
        num_classes=2,
        extractor=Feature_extractor(),
        classifier=Classifier(num_classes=2),
        feature_combination=feature_combination.IdentityFeatureCombination(),
    ).eval()
    inputs = torch.randn(2, 1, 1024)
    gate = torch.full((2, 512), 1.25)

    with torch.no_grad():
        raw = model.extract_raw_features(inputs)
        expected = model.classifier(raw * gate)
        actual = model.classify_features(raw, gate=gate)

    assert torch.allclose(actual, expected)


def test_apply_collectibles_excludes_external_and_red_candidates():
    module = _small_module(source_reports={"source": _report()})
    result = module.apply_collectibles(
        _record(1, 2, 492, 493),
        record_id="r1",
        condition_id="source",
        classifier=_two_collectible_classifier,
    )

    candidate_indices = {entry["index"] for entry in result.diagnostics["candidates"]}
    assert 0 not in candidate_indices
    assert 492 not in candidate_indices
    assert 493 not in candidate_indices
    assert set(result.diagnostics["selected_collectibles"]) == {1, 2}


def test_apply_collectibles_selects_zero_one_or_two_using_k_thresholds():
    module = _small_module(source_reports={"source": _report()})

    none = module.apply_collectibles(
        _record(), "none", condition_id="source", classifier=lambda features: torch.tensor([[0.5, 0.5]] * len(features))
    )
    one = module.apply_collectibles(
        _record(1), "one", condition_id="source", classifier=_one_collectible_classifier
    )
    two = module.apply_collectibles(
        _record(1, 2), "two", condition_id="source", classifier=_two_collectible_classifier
    )

    assert none.diagnostics["selected_collectibles"] == []
    assert one.diagnostics["selected_collectibles"] == [1]
    assert two.diagnostics["selected_collectibles"] == [1, 2]


def test_apply_collectibles_uses_source_quantiles_and_repack_once():
    report = _report(c_quantiles=(0.4, 0.8), delta_quantiles=(0.2, 0.8))
    module = _small_module(source_reports={"source": report})

    result = module.apply_collectibles(
        _record(1),
        record_id="r1",
        condition_id="source",
        classifier=_one_collectible_classifier,
    )

    decision = next(item for item in result.diagnostics["candidates"] if item["index"] == 1)
    assert decision["c_quantiles"] == [0.4, 0.8]
    assert decision["delta_margin_quantiles"] == [0.2, 0.8]
    assert result.diagnostics["repack_count"] == 1
    assert result.diagnostics["repacked"] is True


def test_record_delta_margin_averages_probabilities_before_positive_part():
    module = _small_module(source_reports={"source": _report(c_quantiles=(0.0, 0.8))})
    features = torch.zeros(2, 512)
    features[0, 1] = 1.0
    features[0, 4] = -1.0
    features[1, 1] = -1.0
    features[1, 4] = 1.0

    result = module.apply_collectibles(
        features,
        "mixed",
        condition_id="source",
        classifier=_mixed_sign_margin_classifier,
    )

    decision = next(item for item in result.diagnostics["candidates"] if item["index"] == 1)
    assert decision["delta_clipped"] == 0.0
    assert result.diagnostics["selected_collectibles"] == []


def test_collectible_diagnostics_include_layouts_predictions_and_effective_items():
    module = _small_module(source_reports={"source": _report()})

    result = module.apply_collectibles(
        _record(1),
        "diagnostic",
        condition_id="source",
        classifier=_one_collectible_classifier,
    )

    diagnostics = result.diagnostics
    assert isinstance(diagnostics["initial_layout"], dict)
    assert diagnostics["final_layout"]["selected_indices"] == list(result.layout.selected_indices)
    assert diagnostics["external_features"] == sorted(module.external_indices)
    assert set(diagnostics["red_priority"]) >= {"policy", "applied"}
    detail = next(item for item in diagnostics["selected_item_details"] if item["index"] == 1)
    assert detail["raw_utility"] < detail["effective_utility"]
    assert detail["multiplier"] > 1.0
    assert len(diagnostics["final_window_probabilities"]) == 3
    assert len(diagnostics["final_record_probabilities"]) == 2
    assert diagnostics["final_record_prediction"] == 0


def test_diagnostics_explain_eligible_candidate_dropped_by_two_item_limit():
    keep = {1, 2, 3, 492}
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(),
        external_indices=set(range(512)) - keep,
        source_reports={"source": _report()},
    )

    def three_collectible_classifier(features):
        all_active = (features[:, 1] > 0) & (features[:, 2] > 0) & (features[:, 3] > 0)
        probability = torch.where(all_active, torch.tensor(0.98), torch.tensor(0.5))
        return torch.stack((probability, 1.0 - probability), dim=1)

    result = module.apply_collectibles(
        _record(1, 2, 3), "limit", condition_id="source", classifier=three_collectible_classifier
    )

    dropped = next(item for item in result.diagnostics["candidates"] if item["index"] == 3)
    assert dropped["eligible"] is True
    assert dropped["decision"] == "rejected"
    assert dropped["elimination_reason"] == "only_top_two_collectibles"


def test_direct_model_injection_of_fwc_has_compatible_forward():
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(), external_indices=range(1, 512)
    )
    model = AGGFWCModel(
        num_classes=2,
        extractor=_FixedExtractor(),
        classifier=Classifier(num_classes=2),
        feature_combination=module,
    ).eval()

    with torch.no_grad():
        output = model(torch.zeros(2, 1, 1024), record_ids=["one", "one"])

    assert output.shape == (2, 2)


def test_diagnostics_are_json_serializable_and_candidate_lists_are_independent():
    module = _small_module(source_reports={"source": _report()})

    result = module.apply_collectibles(
        _record(1), "json", condition_id="source", classifier=_one_collectible_classifier
    )

    json.dumps(result.diagnostics)
    assert result.diagnostics["candidates"] is not result.diagnostics["decisions"]
    assert all(left is not right for left, right in zip(result.diagnostics["candidates"], result.diagnostics["decisions"]))
    assert result.diagnostics["candidates"][0]["c_quantiles"] is not result.diagnostics["decisions"][0]["c_quantiles"]


def test_integer_attribute_predictions_are_rejected_before_mean():
    module = feature_combination.FWCFeatureCombination(
        _IntegerAttributePredictor(), external_indices=range(1, 512)
    )

    with pytest.raises(ValueError, match="floating"):
        module.record_gate(torch.zeros(2, 512), record_id="integer-attributes")


@pytest.mark.parametrize("invalid_index", [True, 1.2])
def test_external_indices_reject_bool_and_non_integral_float(invalid_index):
    with pytest.raises(ValueError, match="non-negative integer"):
        feature_combination.FWCFeatureCombination(
            _ConstantAttributePredictor(), external_indices=[invalid_index]
        )


def test_batch_record_gate_groups_records_and_restores_window_order():
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(), external_indices=range(1, 512)
    )
    features = torch.zeros(3, 512)

    result = module.batch_record_gate(features, record_ids=["record-a", "record-b", "record-a"])

    assert result.window_gates.shape == (3, 512)
    assert torch.equal(result.window_gates[0], result.window_gates[2])
    assert tuple(item.record_id for item in result.record_results) == ("record-a", "record-b")


def test_model_requires_record_ids_for_injected_fwc_when_batch_boundaries_are_unknown():
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(), external_indices=range(1, 512)
    )
    model = AGGFWCModel(
        num_classes=2,
        extractor=_FixedExtractor(),
        classifier=Classifier(num_classes=2),
        feature_combination=module,
    ).eval()

    with pytest.raises(ValueError, match="record_ids"):
        model(torch.zeros(2, 1, 1024))


def test_model_extract_features_requires_record_ids_in_fwc_mode():
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(), external_indices=range(1, 512)
    )
    model = AGGFWCModel(
        num_classes=2,
        extractor=_FixedExtractor(),
        classifier=Classifier(num_classes=2),
        feature_combination=module,
    ).eval()

    with pytest.raises(ValueError, match="record_ids"):
        model.extract_features(torch.zeros(2, 1, 1024))


def test_model_extract_features_accepts_record_ids_and_preserves_window_order():
    module = feature_combination.FWCFeatureCombination(
        _ConstantAttributePredictor(), external_indices=range(1, 512)
    )
    model = AGGFWCModel(
        num_classes=2,
        extractor=_FixedExtractor(),
        classifier=Classifier(num_classes=2),
        feature_combination=module,
    ).eval()

    features = model.extract_features(
        torch.zeros(3, 1, 1024), record_ids=["record-a", "record-b", "record-a"]
    )

    assert features.shape == (3, 512)


def test_classifier_output_row_count_is_validated_before_diagnostics():
    module = _small_module(source_reports={"source": _report()})

    with pytest.raises(ValueError, match="rows|batch"):
        module.apply_collectibles(
            _record(1), "bad-classifier", condition_id="source", classifier=lambda features: torch.ones(1, 2)
        )


def test_classifier_output_class_count_must_remain_consistent_across_masking():
    module = _small_module(source_reports={"source": _report()})
    calls = {"count": 0}

    def changing_classifier(features):
        calls["count"] += 1
        class_count = 2 if calls["count"] == 1 else 3
        return torch.ones(features.shape[0], class_count)

    with pytest.raises(ValueError, match="classes"):
        module.apply_collectibles(
            _record(1), "bad-class-count", condition_id="source", classifier=changing_classifier
        )


class _MeanForbiddenProbabilityTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, values):
        return torch.Tensor._make_subclass(cls, values, require_grad=False)

    def mean(self, *args, **kwargs):
        raise AssertionError("class count must be checked before record aggregation")


def test_final_classifier_class_count_is_checked_before_record_probability_mean():
    module = _small_module(source_reports={"source": _report()})
    calls = {"count": 0}

    def changing_final_classifier(features):
        calls["count"] += 1
        if calls["count"] == 1:
            return torch.tensor([[0.5, 0.5]] * len(features))
        if calls["count"] == 2:
            return torch.tensor([[0.5, 0.5]] * len(features))
        return _MeanForbiddenProbabilityTensor(torch.tensor([[1.0, 0.0, 0.0]] * len(features)))

    with pytest.raises(ValueError, match="classes"):
        module.apply_collectibles(
            _record(1), "bad-final-class-count", condition_id="source", classifier=changing_final_classifier
        )


def test_assign_items_rejects_integer_attribute_scores():
    integer_scores = torch.ones((1, 512), dtype=torch.int64)

    with pytest.raises(ValueError, match="floating"):
        assign_items(integer_scores, integer_scores, integer_scores)


def test_source_quantile_dictionary_must_be_bounded_and_ordered():
    module = _small_module(
        source_reports={
            "source": {"c_quantiles": (0.8, 0.2), "delta_margin_quantiles": (0.0, 1.0)}
        }
    )

    with pytest.raises(ValueError, match="quantiles|ordered"):
        module.apply_collectibles(
            _record(1), "bad-quantiles", condition_id="source", classifier=_one_collectible_classifier
        )


def test_source_quantile_dictionary_rejects_values_outside_unit_interval():
    module = _small_module(
        source_reports={
            "source": {"c_quantiles": (-0.1, 0.8), "delta_margin_quantiles": (0.0, 1.0)}
        }
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        module.apply_collectibles(
            _record(1), "out-of-range-quantiles", condition_id="source", classifier=_one_collectible_classifier
        )


def test_equal_scores_512_candidate_exact_pack_finishes_within_ten_seconds():
    script = """
import time
import torch
from AGG_FWC.models.feature_combination import assign_items
from AGG_FWC.models.fwc_packing import exact_pack

scores = torch.ones(1, 512)
items = assign_items(scores, scores, torch.zeros_like(scores))[0]
started = time.perf_counter()
exact_pack(items)
print(time.perf_counter() - started)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("equal-score 512-item exact packing exceeded 10 seconds")

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout.strip()) < 10.0
