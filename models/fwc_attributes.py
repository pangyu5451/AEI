"""Source-only FWC attribute prediction and calibration-label fitting."""

from copy import deepcopy
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from AGG_FWC.models.fwc_calibration import attribute_training_fingerprint
from AGG_FWC.models.fwc_types import ConditionAttributeReport
from AGG_FWC.models.fwc_types import _require_audited_source_payload, _verify_audited_source_payload


class AttributePredictor(nn.Module):
    """Predict contribution, stability, and redundancy from frozen features."""

    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 3),
        )

    def forward(self, source_features):
        if not isinstance(source_features, torch.Tensor):
            raise ValueError("source_features must be a torch.Tensor")
        if source_features.ndim != 2 or source_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"source_features must have shape [B, {self.feature_dim}]"
            )
        c_hat, s_hat, r_hat = torch.chunk(self.network(source_features), 3, dim=1)
        return torch.sigmoid(c_hat), torch.sigmoid(s_hat), torch.sigmoid(r_hat)


def select_source_held_out_records(source_record_ids, held_out_record_id, *, domain="source"):
    """Return train and selection masks for one held-out source record."""
    _validate_source_domain(domain)
    record_ids = tuple(source_record_ids)
    if not record_ids or held_out_record_id not in record_ids:
        raise ValueError("held_out_record_id must identify one source record")
    held_out_mask = torch.tensor(
        [record_id == held_out_record_id for record_id in record_ids], dtype=torch.bool
    )
    if held_out_mask.all():
        raise ValueError("at least one source training record is required")
    return ~held_out_mask, held_out_mask


def fit_attribute_predictor(
    predictor,
    source_features,
    source_condition_ids,
    attribute_reports,
    source_record_ids,
    held_out_record_id,
    audited_source_payload,
    *,
    epochs=1,
    lr=1e-3,
    seed=None,
    domain="source",
    target_features=None,
):
    """Fit on source reports under the trusted-runner audited-payload contract.

    The attestation catches accidental manifest-partition or payload mismatch;
    it is not a defense against hostile code in this Python process.  ``seed``
    controls torch operations performed during fitting; callers seed predictor
    construction separately when they require reproducible initialization.
    """
    _validate_source_domain(domain)
    if target_features is not None:
        raise ValueError("target_features are forbidden during source-only attribute fitting")
    if not isinstance(predictor, AttributePredictor):
        raise ValueError("predictor must be an AttributePredictor")
    if not isinstance(source_features, torch.Tensor) or source_features.requires_grad:
        raise ValueError("source_features must be frozen torch features")
    if not source_features.is_floating_point() or not torch.isfinite(source_features).all():
        raise ValueError("source_features must be a finite floating tensor")
    if source_features.ndim != 2 or source_features.shape[1] != predictor.feature_dim:
        raise ValueError(
            f"source_features must have shape [B, {predictor.feature_dim}]"
        )
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(lr, Real) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a positive finite number")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an integer or None")
    _require_audited_source_payload(audited_source_payload)
    _verify_audited_source_payload(
        audited_source_payload,
        attribute_training_fingerprint(
            source_features, source_condition_ids, source_record_ids, attribute_reports
        ),
    )

    targets = _report_targets(
        source_features, source_condition_ids, attribute_reports, predictor.feature_dim
    )
    train_mask, selection_mask = select_source_held_out_records(
        source_record_ids, held_out_record_id, domain=domain
    )
    if train_mask.numel() != source_features.shape[0]:
        raise ValueError("source features, condition IDs, and record IDs must have matching length")

    original_training = predictor.training
    original_requires_grad = tuple(parameter.requires_grad for parameter in predictor.parameters())
    try:
        for parameter in predictor.parameters():
            parameter.requires_grad_(True)
        predictor.train()
        if seed is not None:
            torch.manual_seed(seed)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=float(lr))
        best_state = None
        best_selection_loss = None
        frozen_features = source_features.detach()
        for _ in range(epochs):
            optimizer.zero_grad()
            train_predictions = predictor(frozen_features[train_mask])
            train_loss = _mse_loss(train_predictions, tuple(target[train_mask] for target in targets))
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                selection_predictions = predictor(frozen_features[selection_mask])
                selection_loss = _mse_loss(
                    selection_predictions, tuple(target[selection_mask] for target in targets)
                )
            if best_selection_loss is None or selection_loss.item() < best_selection_loss:
                best_selection_loss = selection_loss.item()
                best_state = deepcopy(predictor.state_dict())
    except Exception:
        predictor.train(original_training)
        for parameter, requires_grad in zip(predictor.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        raise

    predictor.load_state_dict(best_state)
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor


def _report_targets(source_features, source_condition_ids, attribute_reports, feature_dim):
    if len(source_condition_ids) != source_features.shape[0]:
        raise ValueError("source features and condition IDs must have matching length")
    reports = {}
    for report in attribute_reports:
        if not isinstance(report, ConditionAttributeReport):
            raise ValueError("attribute_reports must contain ConditionAttributeReport values")
        if report.condition_id in reports:
            raise ValueError("attribute report condition IDs must be unique")
        vectors = (report.contribution, report.stability, report.redundancy)
        if any(len(vector) != feature_dim for vector in vectors):
            raise ValueError("attribute report vectors must match feature_dim")
        reports[report.condition_id] = vectors
    if not reports or any(condition_id not in reports for condition_id in source_condition_ids):
        raise ValueError("every source condition ID must have a source attribute report")
    device, dtype = source_features.device, source_features.dtype
    return tuple(
        torch.tensor(
            [reports[condition_id][index] for condition_id in source_condition_ids],
            device=device,
            dtype=dtype,
        )
        for index in range(3)
    )


def _mse_loss(predictions, targets):
    return sum(F.mse_loss(prediction, target) for prediction, target in zip(predictions, targets)) / 3.0


def _validate_source_domain(domain):
    if domain != "source":
        raise ValueError("only the source domain is permitted")
