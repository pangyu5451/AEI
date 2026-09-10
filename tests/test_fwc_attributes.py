import pytest
import torch
import pandas as pd
from dataclasses import replace

from AGG_FWC.config import Config
from AGG_FWC.models.fwc_calibration import create_audited_source_payload
from AGG_FWC.models.fwc_attributes import (
    AttributePredictor,
    fit_attribute_predictor,
    select_source_held_out_records,
)
from AGG_FWC.models.fwc_types import ConditionAttributeReport
from AGG_FWC.protocol import create_audited_source_partition


def _report(condition_id):
    return ConditionAttributeReport(
        condition_id=condition_id,
        contribution=tuple([0.2] * 512),
        stability=tuple([0.6] * 512),
        redundancy=tuple([0.1] * 512),
        c_quantiles=(0.1, 0.3),
        delta_margin_quantiles=(0.0, 0.2),
    )


def _source_training_inputs(tmp_path):
    source_ids = ["source-r0", "source-r0", "source-r1", "source-r1", "source-r2", "source-r2", "source-r3", "source-r3"]
    report = _report("N15_M01_F10")
    source_features = torch.randn(8, 512)
    manifest = _manifest(tmp_path)
    partition = create_audited_source_partition(
        manifest, source_condition_ids=["N15_M01_F10"], target_condition_ids=["TARGET_ONLY"]
    )
    return {
        "source_features": source_features,
        "source_condition_ids": ["N15_M01_F10"] * 8,
        "attribute_reports": [report],
        "source_record_ids": source_ids,
        "held_out_record_id": "source-r3",
        "audited_source_payload": create_audited_source_payload(
            partition, source_features, ["N15_M01_F10"] * 8, source_ids, [report]
        ),
    }


def _manifest(tmp_path):
    rows = []
    for record_id, condition_id in (
        ("source-r0", "N15_M01_F10"),
        ("source-r1", "N15_M01_F10"),
        ("source-r2", "N15_M01_F10"),
        ("source-r3", "N15_M01_F10"),
        ("target-r0", "TARGET_ONLY"),
    ):
        path = tmp_path / f"{record_id}.mat"
        path.write_bytes(record_id.encode("ascii"))
        rows.append(
            {
                "record_id": record_id,
                "condition_id": condition_id,
                "bearing_id": f"bearing-{record_id}",
                "fault_label": "normal",
                "label_provenance": "verified PU metadata",
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def test_attribute_predictor_returns_three_bounded_512_vectors():
    predictor = AttributePredictor()

    c_hat, s_hat, r_hat = predictor(torch.randn(3, 512))

    for attribute in (c_hat, s_hat, r_hat):
        assert attribute.shape == (3, 512)
        assert torch.all((0.0 <= attribute) & (attribute <= 1.0))


def test_attribute_predictor_uses_default_and_requested_hidden_dimensions():
    default_predictor = AttributePredictor()
    compact_predictor = AttributePredictor(hidden_dim=32)

    assert default_predictor.hidden_dim == 128
    assert default_predictor.network[0].in_features == 512
    assert default_predictor.network[0].out_features == 128
    assert default_predictor.network[2].out_features == 1536
    assert compact_predictor.hidden_dim == 32
    assert compact_predictor.network[0].out_features == 32


def test_source_held_out_record_selection_rejects_non_source_markers():
    with pytest.raises(ValueError, match="source"):
        select_source_held_out_records(["r0", "r0", "r1", "r1"], "r1", domain="target")


def test_attribute_training_rejects_target_input_and_non_source_marker(tmp_path):
    inputs = _source_training_inputs(tmp_path)

    with pytest.raises(ValueError, match="target"):
        fit_attribute_predictor(AttributePredictor(), **inputs, target_features=torch.randn(2, 512))
    with pytest.raises(ValueError, match="source"):
        fit_attribute_predictor(AttributePredictor(), **inputs, domain="target")


def test_attribute_training_uses_source_reports_and_freezes_predictor(tmp_path):
    predictor = AttributePredictor(hidden_dim=16)

    fitted = fit_attribute_predictor(predictor, **_source_training_inputs(tmp_path), epochs=2, lr=1e-3)

    assert fitted is predictor
    assert not predictor.training
    assert all(not parameter.requires_grad for parameter in predictor.parameters())


def test_attribute_training_rejects_target_only_bypass_despite_source_domain_marker(tmp_path):
    inputs = _source_training_inputs(tmp_path)
    target_features = torch.randn(4, 512)
    target_report = _report("TARGET_ONLY")

    with pytest.raises(ValueError, match="attestation"):
        fit_attribute_predictor(
            AttributePredictor(),
            source_features=target_features,
            source_condition_ids=["TARGET_ONLY"] * 4,
            attribute_reports=[target_report],
            source_record_ids=["target-r0", "target-r0", "target-r1", "target-r1"],
            held_out_record_id="target-r1",
            audited_source_payload=inputs["audited_source_payload"],
            domain="source",
        )


@pytest.mark.parametrize(
    "field_name, replacement",
    [("c_quantiles", (0.15, 0.3)), ("delta_margin_quantiles", (0.01, 0.2))],
)
def test_attribute_training_rejects_report_quantile_change_after_attestation(
    tmp_path, field_name, replacement
):
    inputs = _source_training_inputs(tmp_path)
    changed_report = replace(inputs["attribute_reports"][0], **{field_name: replacement})

    with pytest.raises(ValueError, match="attestation"):
        fit_attribute_predictor(
            AttributePredictor(),
            **{**inputs, "attribute_reports": [changed_report]},
        )


def test_attribute_training_restores_held_out_best_checkpoint(tmp_path):
    source_features = torch.zeros(8, 1)
    source_ids = ["r0", "r0", "r1", "r1", "r2", "r2", "r3", "r3"]
    source_conditions = ["A"] * 6 + ["B"] * 2
    reports = [
        ConditionAttributeReport("A", (0.0,), (0.0,), (0.0,), (0.0, 0.0), (0.0, 0.0)),
        ConditionAttributeReport("B", (1.0,), (1.0,), (1.0,), (1.0, 1.0), (0.0, 0.0)),
    ]
    manifest = _small_manifest(tmp_path)
    partition = create_audited_source_partition(manifest, ["A", "B"], ["TARGET_ONLY"])
    inputs = dict(
        source_features=source_features,
        source_condition_ids=source_conditions,
        attribute_reports=reports,
        source_record_ids=source_ids,
        held_out_record_id="r3",
        audited_source_payload=create_audited_source_payload(
            partition, source_features, source_conditions, source_ids, reports
        ),
        lr=0.1,
        seed=7,
    )
    first_epoch = AttributePredictor(feature_dim=1, hidden_dim=1)
    two_epochs = AttributePredictor(feature_dim=1, hidden_dim=1)
    for predictor in (first_epoch, two_epochs):
        for parameter in predictor.parameters():
            parameter.data.zero_()

    fit_attribute_predictor(first_epoch, **inputs, epochs=1)
    fit_attribute_predictor(two_epochs, **inputs, epochs=2)

    for first, restored in zip(first_epoch.parameters(), two_epochs.parameters()):
        assert torch.allclose(first, restored)


@pytest.mark.parametrize("lr", [float("nan"), float("inf"), float("-inf")])
def test_attribute_training_rejects_nonfinite_lr_without_predictor_state_mutation(tmp_path, lr):
    predictor = _frozen_eval_predictor()
    inputs = _source_training_inputs(tmp_path)

    with pytest.raises(ValueError, match="finite"):
        fit_attribute_predictor(predictor, **inputs, lr=lr)

    _assert_frozen_eval(predictor)
    assert all(torch.isfinite(parameter).all() for parameter in predictor.parameters())


def test_attribute_training_rejects_integer_features_without_predictor_state_mutation(tmp_path):
    predictor = _frozen_eval_predictor()
    inputs = _source_training_inputs(tmp_path)
    integer_features = torch.zeros_like(inputs["source_features"], dtype=torch.int64)
    partition = create_audited_source_partition(
        _manifest(tmp_path), ["N15_M01_F10"], ["TARGET_ONLY"]
    )
    integer_payload = create_audited_source_payload(
        partition,
        integer_features,
        inputs["source_condition_ids"],
        inputs["source_record_ids"],
        inputs["attribute_reports"],
    )

    with pytest.raises(ValueError, match="floating"):
        fit_attribute_predictor(
            predictor,
            **{
                **inputs,
                "source_features": integer_features,
                "audited_source_payload": integer_payload,
            },
        )

    _assert_frozen_eval(predictor)


def test_attribute_training_restores_predictor_state_after_training_phase_error(tmp_path):
    predictor = _frozen_eval_predictor().double()
    inputs = _source_training_inputs(tmp_path)

    with pytest.raises(RuntimeError):
        fit_attribute_predictor(predictor, **inputs)

    _assert_frozen_eval(predictor)


def _frozen_eval_predictor():
    predictor = AttributePredictor()
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor


def _assert_frozen_eval(predictor):
    assert not predictor.training
    assert all(not parameter.requires_grad for parameter in predictor.parameters())


def _small_manifest(tmp_path):
    rows = []
    for record_id, condition_id in (("r0", "A"), ("r1", "A"), ("r2", "A"), ("r3", "B"), ("target", "TARGET_ONLY")):
        path = tmp_path / f"{record_id}.mat"
        path.write_bytes(record_id.encode("ascii"))
        rows.append({"record_id": record_id, "condition_id": condition_id, "bearing_id": f"bearing-{record_id}", "fault_label": "normal", "label_provenance": "verified", "path": str(path)})
    return pd.DataFrame(rows)


def test_config_exposes_reproducible_fwc_attribute_defaults():
    config = Config()

    assert isinstance(config.fwc_stage, str) and config.fwc_stage
    assert config.attribute_hidden_dim == 128
    assert isinstance(config.attribute_epoch, int) and config.attribute_epoch > 0
    assert isinstance(config.attribute_lr, float) and config.attribute_lr > 0.0
