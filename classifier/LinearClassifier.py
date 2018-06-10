import numpy as np
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from Settings import N_JOBS
from Utils import clamp, inherit_docstrings
from classifier.Classifier import Classifier


@inherit_docstrings
class LinearClassifier(Classifier):
    """
    A simple linear SVM classifier.
    """

    def __init__(self, c: float = 1.0, verbose: int = 0, max_iter: int = 1000) -> None:
        """
        Instantiate a Linear SVM classifier with the given parameters.
        Will be overridden if load is called later.
        :param c: The penalty parameter
        :param verbose: The level of logging
        :param max_iter: The maximum number of iterations to converge
        """
        super().__init__(verbose)
        self.c = c
        self.classifier = LinearSVC(C=c, verbose=verbose, max_iter=max_iter)

    def cross_validate(self, cv: int, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        scores = cross_val_score(self.classifier, features, labels, cv=cv, n_jobs=N_JOBS, verbose=self.verbose)
        return scores

    def get_classifier_name(self) -> str:
        return "LinearClassifier - C " + str(self.c)

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

    def reset(self) -> None:
        self.classifier = clone(self.classifier)
