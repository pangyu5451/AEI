import torch

from AGG_FWC.models.agg_fwc_model import AGGFWCModel
from AGG_FWC.models.classifier import Classifier
from AGG_FWC.models.extractor import Feature_extractor
from AGG_FWC.models.feature_combination import IdentityFeatureCombination


def test_identity_path_matches_original_extractor_and_classifier():
    torch.manual_seed(2026)
    extractor = Feature_extractor()
    classifier = Classifier(num_classes=14)
    model = AGGFWCModel(
        num_classes=14,
        extractor=extractor,
        classifier=classifier,
        feature_combination=IdentityFeatureCombination(),
    )

    inputs = torch.randn(2, 1, 1024, requires_grad=True)
    extractor.eval()
    classifier.eval()
    model.eval()

    with torch.no_grad():
        original_features = extractor(inputs)
        original_logits = classifier(original_features)
        features = model.extract_features(inputs)
        logits = model(inputs)

    assert features.shape == (2, 512)
    assert logits.shape == (2, 14)
    assert torch.allclose(features, original_features)
    assert torch.allclose(logits, original_logits)

    model(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
