import numpy as np

from Settings import SEED, N_JOBS
from classifier.Classifier import Classifier

np.random.seed(seed=SEED)

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from Utils import inherit_docstrings
from sklearn.model_selection import cross_val_score
from sklearn.base import clone


@inherit_docstrings
class RFClassifier(Classifier):
    """
    A Random Forest Classifier.
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = None, seed: int = SEED, verbose: int = 0,
                 n_jobs: int = N_JOBS) -> None:
        """
        Instantiate a RandomForest Classifier with the given parameters.
        Will be overridden if load is called later.
        :param n_estimators: The number of trees to build
        :param max_depth: The maximum depth of the trees
        :param seed: The seed to use for the random operations
        :param verbose: The level of logging
        :param n_jobs: The number of threads to use to build the forest
        """
        super().__init__(verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
                                                 verbose=verbose, n_jobs=n_jobs)

    def reset(self) -> None:
        self.classifier = clone(self.classifier)

    def cross_validate(self, cv: int, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        scores = cross_val_score(self.classifier, features, labels, cv=cv, n_jobs=1, verbose=self.verbose)
        return scores

    def get_classifier_name(self) -> str:
        return "RFClassifier - n_est " + str(self.n_estimators) + " - max_depth " + str(self.max_depth)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(features)

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
