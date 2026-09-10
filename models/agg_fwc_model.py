import torch.nn as nn

from AGG_FWC.models.classifier import Classifier
from AGG_FWC.models.extractor import Feature_extractor
from AGG_FWC.models.feature_combination import FWCFeatureCombination, IdentityFeatureCombination


class AGGFWCModel(nn.Module):
    def __init__(
        self,
        num_classes,
        extractor=None,
        classifier=None,
        feature_combination=None,
    ):
        super().__init__()
        self.extractor = extractor if extractor is not None else Feature_extractor()
        self.feature_combination = (
            feature_combination
            if feature_combination is not None
            else IdentityFeatureCombination()
        )
        self.classifier = classifier if classifier is not None else Classifier(num_classes)

    def extract_features(self, inputs, record_ids=None):
        features = self.extract_raw_features(inputs)
        if isinstance(self.feature_combination, FWCFeatureCombination):
            if record_ids is None:
                raise ValueError(
                    "record_ids are required when FWC receives a possibly mixed record batch"
                )
            if isinstance(record_ids, (str, int)):
                record_ids = [record_ids] * features.shape[0]
            batch_result = self.feature_combination.batch_record_gate(features, record_ids)
            return features * batch_result.window_gates
        return self.feature_combination(features)

    def extract_raw_features(self, inputs):
        return self.extractor(inputs)

    def classify_features(self, features, gate=None):
        if gate is not None:
            features = features * gate
        return self.classifier(features)

    def forward(self, inputs, record_ids=None):
        return self.classify_features(self.extract_features(inputs, record_ids=record_ids))
