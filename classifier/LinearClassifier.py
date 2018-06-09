import numpy as np
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from Utils import clamp, inherit_docstrings
from classifier.Classifier import Classifier


@inherit_docstrings
class LinearClassifier(Classifier):
    """
    A simple linear SVM classifier
    """
    classifier: LinearSVC

    def __init__(self, verbose: int = 0) -> None:
        """
        Instantiates a Linear SVM classifier
        :param verbose: The level of logging to display
        """
        self.classifier = LinearSVC(C=1, verbose=verbose, max_iter=10000)

    def get_classifier_name(self) -> str:
        return "LinearClassifier"

    def predict(self, features: np.ndarray) -> np.ndarray:
        return clamp(self.classifier.predict(features))

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.classifier.fit(features, labels)

    def save(self, filename: str) -> None:
        joblib.dump(self.classifier, filename, compress=5)

    def load(self, filename: str) -> bool:
        if self.check_dump_exists(filename):
            self.classifier = joblib.load(filename)
            return True
        else:
            return False
