from classifier.Classifier import Classifier
import numpy as np
from Settings import M_LABEL


class ConstantClassifier(Classifier):
    def get_classifier_name(self) -> str:
        return "ConstantClassifier"

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.array(list(map(lambda f: M_LABEL, features)))

    def train(self, features: np.ndarray, labels: np.ndarray, cv: int = None) -> None:
        pass

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> bool:
        return True
