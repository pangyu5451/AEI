import inspect

import numpy as np
import pytest

from AGG_FWC.models.fwc_calibration import (
    _grouped_oof_predictions,
    _resample_groups,
    calibrate_condition,
    load_condition_attribute_report,
    save_condition_attribute_report,
    select_external_features,
    single_feature_grouped_oof_accuracies,
    utility,
)
from AGG_FWC.models.fwc_types import (
    ConditionAttributeReport,
    Item,
    Layout,
    Placement,
)


def _source_data():
    record_ids = np.repeat(np.array(["r0", "r1", "r2", "r3", "r4", "r5"]), 4)
    labels = np.tile(np.array([0, 0, 1, 1]), 6)
    signal = np.tile(np.array([-2.0, -1.6, 1.6, 2.0]), 6)
    noise = np.array(
        [
            0.10,
            -0.20,
            0.30,
            -0.10,
            -0.30,
            0.20,
            -0.10,
            0.30,
            0.20,
            0.10,
            -0.30,
            -0.20,
            0.30,
            -0.10,
            0.20,
            -0.20,
            -0.10,
            0.30,
            0.10,
            -0.30,
            0.20,
            -0.20,
            0.30,
            -0.10,
        ]
    )
    return np.column_stack([signal, noise]), labels, record_ids


def _imbalanced_group_source():
    record_ids = np.repeat(np.array(["g0", "g1", "g2", "g3", "g4", "g5"]), 2)
    labels = np.repeat(np.array([0, 0, 0, 0, 1, 1]), 2)
    features = np.column_stack([2.0 * labels - 1.0, np.arange(labels.size, dtype=float)])
    return features, labels, record_ids


def _four_group_source():
    record_ids = np.repeat(np.array(["g0", "g1", "g2", "g3"]), 2)
    labels = np.tile(np.array([0, 1]), 4)
    features = np.column_stack([2.0 * labels - 1.0, np.arange(labels.size, dtype=float)])
    return features, labels, record_ids


def test_fwc_types_are_immutable_calibration_records():
    item = Item(index=3, color="blue", size=2, utility=0.75)
    placement = Placement(item_index=3, mask=0b11, shape=(1, 2))
    layout = Layout((3,), (placement,), 0.75, 2, False)

    assert item.c == item.s == item.r == 0.0
    assert layout.placements == (placement,)
    with pytest.raises(Exception):
        item.utility = 0.1


def test_condition_attribute_report_exposes_only_immutable_vectors():
    contribution = np.array([0.4, 0.2])
    c_quantiles = np.array([0.21, 0.39])
    delta_margin_quantiles = np.array([0.02, 0.20])
    report = ConditionAttributeReport(
        condition_id="N15_M01_F10",
        contribution=contribution,
        stability=np.array([0.8, 0.6]),
        redundancy=np.array([0.0, 0.3]),
        c_quantiles=c_quantiles,
        delta_margin_quantiles=delta_margin_quantiles,
    )

    contribution[0] = 9.0
    c_quantiles[0] = 9.0
    delta_margin_quantiles[0] = 9.0

    assert report.contribution == (0.4, 0.2)
    assert report.c_quantiles == (0.21, 0.39)
    assert report.delta_margin_quantiles == (0.02, 0.20)
    for vector in (
        report.contribution,
        report.stability,
        report.redundancy,
        report.c_quantiles,
        report.delta_margin_quantiles,
    ):
        assert isinstance(vector, tuple)
        assert not hasattr(vector, "setflags")
        with pytest.raises(TypeError):
            vector[0] = 0.1


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_condition_attribute_report_rejects_nonfinite_or_out_of_range_attributes(value):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        ConditionAttributeReport(
            condition_id="N15_M01_F10",
            contribution=(value, 0.2),
            stability=(0.8, 0.6),
            redundancy=(0.0, 0.3),
            c_quantiles=(0.2, 0.4),
            delta_margin_quantiles=(0.0, 0.2),
        )


def test_utility_clips_each_attribute_to_unit_interval():
    result = utility(
        np.array([-0.2, 0.5, 1.5]),
        np.array([1.2, 0.5, 0.5]),
        np.array([0.1, -0.5, 2.0]),
    )

    np.testing.assert_allclose(result, [0.0, 0.25, 0.0])


def test_attribute_calibration_prefers_stable_nonredundant_contributor():
    features, labels, record_ids = _source_data()

    report = calibrate_condition(features, labels, record_ids, repeats=5, seed=2026)

    assert isinstance(report, ConditionAttributeReport)
    assert report.condition_id == "source"
    assert report.contribution[0] > report.contribution[1]
    assert np.all((0.0 <= np.asarray(report.contribution)) & (np.asarray(report.contribution) <= 1.0))
    assert report.c_quantiles == pytest.approx(np.quantile(report.contribution, [0.05, 0.95]))
    assert report.redundancy[0] == pytest.approx(0.0)
    stability = np.asarray(report.stability)
    assert np.all((0.0 <= stability) & (stability <= 1.0))
    assert len(report.c_quantiles) == len(report.delta_margin_quantiles) == 2
    assert np.isfinite(np.asarray(report.c_quantiles)).all()
    assert np.isfinite(np.asarray(report.delta_margin_quantiles)).all()


def test_calibration_report_round_trip_uses_safe_numeric_npz_arrays(tmp_path):
    features, labels, record_ids = _source_data()
    report = calibrate_condition(features, labels, record_ids)
    output_path = tmp_path / "source-report.npz"

    saved_path = save_condition_attribute_report(report, output_path)
    loaded = load_condition_attribute_report(output_path)

    assert saved_path == output_path
    with np.load(output_path, allow_pickle=False) as archive:
        assert all(values.dtype != object for values in archive.values())
    assert loaded.condition_id == report.condition_id
    np.testing.assert_allclose(loaded.contribution, report.contribution)
    np.testing.assert_allclose(loaded.stability, report.stability)
    np.testing.assert_allclose(loaded.redundancy, report.redundancy)
    assert loaded.c_quantiles == pytest.approx(report.c_quantiles)
    assert loaded.delta_margin_quantiles == pytest.approx(report.delta_margin_quantiles)


def test_save_report_normalizes_extensionless_output_before_immediate_load(tmp_path):
    features, labels, record_ids = _source_data()
    report = calibrate_condition(features, labels, record_ids)
    requested_path = tmp_path / "extensionless-report"

    saved_path = save_condition_attribute_report(report, requested_path)
    loaded = load_condition_attribute_report(saved_path)

    assert saved_path == requested_path.with_suffix(".npz")
    assert saved_path.exists()
    assert loaded == report


def test_save_report_normalizes_non_npz_suffix_before_immediate_load(tmp_path):
    features, labels, record_ids = _source_data()
    report = calibrate_condition(features, labels, record_ids)
    requested_path = tmp_path / "report.json"

    saved_path = save_condition_attribute_report(report, requested_path)
    loaded = load_condition_attribute_report(saved_path)

    assert saved_path == requested_path.with_suffix(".npz")
    assert saved_path.exists()
    assert loaded == report


def test_calibration_preserves_explicit_condition_id_through_save_load(tmp_path):
    features, labels, record_ids = _source_data()
    report = calibrate_condition(
        features, labels, record_ids, condition_id="N15_M01_F10"
    )

    save_condition_attribute_report(report, tmp_path / "condition-report.npz")
    loaded = load_condition_attribute_report(tmp_path / "condition-report.npz")

    assert report.condition_id == loaded.condition_id == "N15_M01_F10"


@pytest.mark.parametrize("condition_id", ["", "   ", None])
def test_calibration_rejects_blank_condition_id(condition_id):
    features, labels, record_ids = _source_data()

    with pytest.raises(ValueError, match="condition_id.*non-empty"):
        calibrate_condition(features, labels, record_ids, condition_id=condition_id)


def test_select_external_features_uses_both_inclusive_thresholds_and_allows_empty():
    accuracies = np.array(
        [
            [0.85, 0.95, 0.96, 0.84],
            [0.85, 0.75, 0.74, 0.84],
            [0.85, 0.82, 0.85, 0.84],
        ]
    )

    selected = select_external_features(accuracies)

    np.testing.assert_array_equal(selected, [False, False, True, False])
    np.testing.assert_array_equal(select_external_features(np.full((3, 2), 0.84)), [False, False])


def test_single_feature_grouped_oof_accuracies_are_source_only_and_discriminative():
    features, labels, record_ids = _source_data()

    accuracies = single_feature_grouped_oof_accuracies(features, labels, record_ids)

    assert accuracies.shape == (2,)
    assert accuracies[0] > accuracies[1]
    assert np.all((0.0 <= accuracies) & (accuracies <= 1.0))


def test_grouped_oof_rejects_fold_without_all_training_classes():
    features = np.array([[-1.0], [-0.8], [0.8], [1.0]])
    labels = np.array([0, 0, 1, 1])
    record_ids = np.array(["class-zero", "class-zero", "class-one", "class-one"])

    with pytest.raises(ValueError, match="grouped-fold class coverage"):
        single_feature_grouped_oof_accuracies(features, labels, record_ids)


def test_stability_completes_five_valid_grouped_resamples_for_imbalanced_source():
    features, labels, record_ids = _imbalanced_group_source()

    for resample_index in range(5):
        resampled_features, resampled_labels, resampled_ids = _resample_groups(
            features, labels, record_ids, seed=2026 + resample_index
        )
        _grouped_oof_predictions(
            resampled_features, resampled_labels, resampled_ids, seed=2026 + resample_index
        )

    report = calibrate_condition(features, labels, record_ids, seed=2026)
    assert np.asarray(report.stability).shape == (2,)


def test_four_group_stability_resample_omits_at_least_one_group():
    features, labels, record_ids = _four_group_source()

    _, _, resampled_ids = _resample_groups(features, labels, record_ids, seed=2026)

    assert np.unique(resampled_ids).size == 3


def test_four_group_resampling_fails_when_no_omitting_subset_has_oof_coverage():
    record_ids = np.repeat(np.array(["g0", "g1", "g2", "g3"]), 2)
    labels = np.repeat(np.array([0, 0, 1, 1]), 2)
    features = np.column_stack([2.0 * labels - 1.0, np.arange(labels.size, dtype=float)])

    with pytest.raises(ValueError, match="no valid grouped OOF resample"):
        _resample_groups(features, labels, record_ids, seed=2026)


@pytest.mark.parametrize(
    "function, args",
    [
        (calibrate_condition, lambda: _source_data()),
        (single_feature_grouped_oof_accuracies, lambda: _source_data()),
    ],
)
def test_source_only_interfaces_reject_target_data(function, args):
    features, labels, record_ids = args()

    assert not any("target" in name for name in inspect.signature(function).parameters)
    with pytest.raises(TypeError):
        function(features, labels, record_ids, target_features=features)


@pytest.mark.parametrize(
    "features, labels, record_ids, repeats, message",
    [
        (np.ones(4), np.array([0, 1, 0, 1]), np.array(["a", "a", "b", "b"]), 5, "two-dimensional"),
        (np.ones((4, 2)), np.array([0, 1]), np.array(["a", "a", "b", "b"]), 5, "same length"),
        (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array(["a", "b"]), 5, "same length"),
        (np.array([[1.0, np.nan], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), np.array([0, 1, 0, 1]), np.array(["a", "a", "b", "b"]), 5, "finite"),
        (np.ones((4, 2)), np.array([0, 0, 0, 0]), np.array(["a", "a", "b", "b"]), 5, "at least two classes"),
        (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array(["a", "a", "a", "a"]), 5, "at least two groups"),
        (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array(["a", "a", "b", "b"]), 4, "exactly five"),
    ],
)
def test_calibration_rejects_malformed_or_non_source_inputs(
    features, labels, record_ids, repeats, message
):
    with pytest.raises(ValueError, match=message):
        calibrate_condition(features, labels, record_ids, repeats=repeats)


@pytest.mark.parametrize(
    "accuracies",
    [
        np.array([0.95, 0.95]),
        np.array([[0.95, np.nan]]),
        np.empty((0, 2)),
        np.array([[-0.01, 0.95]]),
        np.array([[0.95, 1.01]]),
    ],
)
def test_external_selection_rejects_malformed_accuracy_tables(accuracies):
    with pytest.raises(ValueError):
        select_external_features(accuracies)


def test_calibration_wraps_mixed_type_record_id_sorting_failure():
    features = np.array([[-1.0], [-0.8], [0.8], [1.0]])
    labels = np.array([0, 0, 1, 1])
    record_ids = np.array([0, "g1", 2, "g3"], dtype=object)

    with pytest.raises(ValueError, match="record IDs.*comparable"):
        calibrate_condition(features, labels, record_ids)
