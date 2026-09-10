import pytest
import torch

from AGG_FWC.models import feature_combination
from AGG_FWC.models.feature_combination import IdentityFeatureCombination
from AGG_FWC.models.fwc_types import Item


def test_identity_feature_combination_preserves_values_shape_and_gradients():
    features = torch.randn(4, 512, requires_grad=True)
    combined = IdentityFeatureCombination()(features)

    assert combined.shape == features.shape
    assert torch.equal(combined, features)

    combined.square().mean().backward()

    assert features.grad is not None
    expected_gradient = 2 * features.detach() / features.numel()
    assert torch.allclose(features.grad, expected_gradient)
    assert torch.isfinite(features.grad).all()


def _scores():
    contribution = torch.ones(1, 512)
    stability = torch.ones_like(contribution)
    redundancy = torch.full_like(contribution, 0.25)
    return contribution, stability, redundancy


def test_feature_combination_exposes_task4_assignment_interface():
    assert hasattr(feature_combination, "LEGAL_SIZES")
    assert hasattr(feature_combination, "assign_items")


def test_assign_items_uses_fixed_palette_legal_sizes_and_data_driven_utility():
    contribution, stability, redundancy = _scores()
    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert len(items) == 512
    assert all(isinstance(item, Item) for item in items)
    assert {item.color for item in items} == set(feature_combination.LEGAL_SIZES)
    assert {color: sum(item.color == color for item in items) for color in feature_combination.LEGAL_SIZES} == {
        "gray": 154,
        "green": 128,
        "blue": 102,
        "purple": 67,
        "gold": 41,
        "red": 20,
    }
    assert all(item.size in feature_combination.LEGAL_SIZES[item.color] for item in items)
    assert all(item.utility == pytest.approx(item.c * item.s * (1.0 - item.r)) for item in items)
    assert items[0].color == "gray"
    assert items[-1].color == "red"


def test_assign_items_downgrades_ineligible_red_candidates_to_gold():
    contribution, stability, redundancy = _scores()
    stability[:, -20:] = 0.0
    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert sum(item.color == "red" for item in items) == 0
    assert sum(item.color == "gold" for item in items) == 61
    assert all(item.color == "gold" for item in items[-20:])


def test_assign_items_has_no_red_when_quality_threshold_never_passes():
    contribution = torch.zeros(1, 512)
    stability = torch.zeros(1, 512)
    redundancy = torch.ones(1, 512)

    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert sum(item.color == "red" for item in items) == 0
    assert sum(item.color == "gold" for item in items) == 61


def test_assign_items_breaks_all_contribution_ties_by_feature_index():
    contribution = torch.full((1, 512), 0.9)
    stability = torch.ones_like(contribution)
    redundancy = torch.full_like(contribution, 0.1)

    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert [item.color for item in items[:154]] == ["gray"] * 154
    assert [item.color for item in items[154:282]] == ["green"] * 128
    assert [item.color for item in items[-20:]] == ["red"] * 20


def test_assign_items_reverses_red_size_preference_at_median_equality():
    contribution, stability, redundancy = _scores()
    stability[:, -20:] = 1.0
    redundancy[:, -20:] = 0.5
    redundancy[0, -20] = 0.0
    stability[0, -20] = 0.5

    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert items[-20].color == "red"
    assert items[-1].color == "red"
    assert items[-20].utility == pytest.approx(items[-1].utility)
    assert items[-20].size > items[-1].size


def test_assign_items_is_repeatable_and_rejects_invalid_input():
    contribution, stability, redundancy = _scores()

    assert feature_combination.assign_items(contribution, stability, redundancy) == feature_combination.assign_items(
        contribution, stability, redundancy
    )

    with pytest.raises(ValueError):
        feature_combination.assign_items(torch.ones(512), stability, redundancy)
    with pytest.raises(ValueError):
        feature_combination.assign_items(torch.full((1, 512), float("nan")), stability, redundancy)
    with pytest.raises(ValueError):
        feature_combination.assign_items(torch.full((1, 512), 1.01), stability, redundancy)


def test_public_legal_sizes_cannot_mutate_assignment_rules():
    contribution, stability, redundancy = _scores()
    expected = feature_combination.assign_items(contribution, stability, redundancy)

    with pytest.raises(TypeError):
        feature_combination.LEGAL_SIZES["red"] = (99,)

    assert feature_combination.assign_items(contribution, stability, redundancy) == expected


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_assign_items_keeps_exact_red_quality_boundary_across_float_dtypes(dtype):
    contribution = torch.full((1, 512), 0.63, dtype=dtype)
    stability = torch.full((1, 512), 0.95, dtype=dtype)
    redundancy = torch.zeros((1, 512), dtype=dtype)

    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert sum(item.color == "red" for item in items) == 20


def test_assign_items_rejects_clearly_below_red_quality_boundary():
    contribution = torch.full((1, 512), 0.62)
    stability = torch.full((1, 512), 0.95)
    redundancy = torch.zeros((1, 512))

    items = feature_combination.assign_items(contribution, stability, redundancy)[0]

    assert sum(item.color == "red" for item in items) == 0
