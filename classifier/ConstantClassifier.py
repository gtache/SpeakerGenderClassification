import numpy as np
from sklearn.model_selection import StratifiedKFold

from Settings import M_LABEL, SEED
from Utils import inherit_docstrings, get_accuracy
from classifier.Classifier import Classifier


@inherit_docstrings
class ConstantClassifier(Classifier):
    """
    A classifier predicting Male for every input.
    """

    def __init__(self):
        super().__init__()

    def cross_validate(self, cv: int, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        skf = StratifiedKFold(n_splits=cv, random_state=SEED)
        scores = []
        for train_idx, test_idx in skf.split(features, labels):
            scores.append(get_accuracy(self.predict(features[test_idx]), labels[test_idx]))
        return np.asarray(scores)

    def get_classifier_name(self) -> str:
        return "ConstantClassifier"

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array(list(map(lambda f: M_LABEL, features)))

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        pass

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> bool:
        return True

    def reset(self) -> None:
        pass
