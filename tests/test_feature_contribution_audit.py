import inspect

import numpy as np
import pytest
import torch
from torch import nn

from AGG_FWC.models.feature_contribution_audit import (
    audit_source_feature_contributions,
    load_feature_contribution_audit,
    save_feature_contribution_audit,
)
from AGG_FWC.models import audit_source_feature_contributions as public_audit


def _source_data():
    features = np.array(
        [
            [-1.0, 0.1],
            [-1.0, -0.2],
            [1.0, 0.0],
            [-1.0, 0.3],
            [-1.0, -0.1],
            [1.0, 0.2],
            [1.0, -0.2],
            [1.0, 0.1],
            [-1.0, 0.0],
            [1.0, 0.2],
            [1.0, -0.1],
            [-1.0, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    record_ids = np.repeat(np.array(["r2", "r1", "r4", "r3"]), 3)
    condition_ids = np.repeat(np.array(["z", "z", "a", "a"]), 3)
    return features, labels, record_ids, condition_ids


def _classifier():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-3.0, 0.0], [3.0, 0.0]]))
    return model


def test_audit_aggregates_windows_to_records_and_ranks_informative_feature():
    features, labels, record_ids, condition_ids = _source_data()
    model = _classifier()
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    report = audit_source_feature_contributions(
        model,
        features,
        labels,
        record_ids,
        condition_ids,
        device="cpu",
        seed=2026,
        batch_size=4,
    )

    assert report.feature_count == 2
    assert report.record_count_by_condition == (2, 2)
    assert report.condition_ids == ("a", "z")
    assert report.global_nll_increase[0] > report.global_nll_increase[1]
    assert report.global_macro_f1_drop[0] > report.global_macro_f1_drop[1]
    assert report.global_macro_f1_drop[0] > 0.0
    assert model.training is True
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_audit_honors_reference_and_is_deterministic():
    features, labels, record_ids, condition_ids = _source_data()
    reference = np.array([0.25, 0.75], dtype=np.float32)

    first = audit_source_feature_contributions(
        _classifier(), features, labels, record_ids, condition_ids, device="cpu", reference=reference
    )
    second = audit_source_feature_contributions(
        _classifier(), features, labels, record_ids, condition_ids, device="cpu", reference=reference
    )

    assert first.reference == pytest.approx(tuple(reference))
    assert first == second


def test_audit_signature_has_no_target_argument():
    assert "target" not in inspect.signature(audit_source_feature_contributions).parameters
    assert public_audit is audit_source_feature_contributions


def test_audit_report_round_trip_uses_numeric_arrays(tmp_path):
    features, labels, record_ids, condition_ids = _source_data()
    report = audit_source_feature_contributions(
        _classifier(), features, labels, record_ids, condition_ids, device="cpu"
    )
    saved = save_feature_contribution_audit(report, tmp_path / "audit")
    loaded = load_feature_contribution_audit(saved)

    assert saved == tmp_path / "audit.npz"
    assert loaded == report
    with np.load(saved, allow_pickle=False) as archive:
        assert all(value.dtype != object for value in archive.values())
    assert (tmp_path / "audit.json").exists()


def test_audit_persistence_refuses_overwrite(tmp_path):
    features, labels, record_ids, condition_ids = _source_data()
    report = audit_source_feature_contributions(
        _classifier(), features, labels, record_ids, condition_ids, device="cpu"
    )
    save_feature_contribution_audit(report, tmp_path / "audit")

    with pytest.raises(FileExistsError):
        save_feature_contribution_audit(report, tmp_path / "audit")


@pytest.mark.parametrize(
    "features, labels, record_ids, condition_ids, match",
    [
        (np.ones(4), np.array([0, 1, 0, 1]), np.array(["a"] * 4), np.array(["c"] * 4), "two-dimensional"),
        (np.ones((4, 2)), np.array([0, 1]), np.array(["a"] * 4), np.array(["c"] * 4), "same length"),
        (np.array([[1.0, np.nan]] * 4), np.array([0, 1, 0, 1]), np.array(["a"] * 4), np.array(["c"] * 4), "finite"),
        (np.ones((4, 2)), np.array([0.0, 1.0, 0.0, 1.0]), np.array(["a"] * 4), np.array(["c"] * 4), "integer"),
        (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array(["a"] * 4), np.array(["c"] * 4), "records"),
    ],
)
def test_audit_rejects_malformed_source_arrays(
    features, labels, record_ids, condition_ids, match
):
    with pytest.raises(ValueError, match=match):
        audit_source_feature_contributions(
            _classifier(), features, labels, record_ids, condition_ids, device="cpu"
        )


def test_audit_rejects_bad_reference():
    features, labels, record_ids, condition_ids = _source_data()

    with pytest.raises(ValueError, match="reference"):
        audit_source_feature_contributions(
            _classifier(),
            features,
            labels,
            record_ids,
            condition_ids,
            device="cpu",
            reference=np.zeros(3),
        )
