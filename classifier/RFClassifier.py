import numpy as np

from Settings import SEED
from classifier.Classifier import Classifier

np.random.seed(seed=SEED)

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import multiprocessing
from Utils import inherit_docstrings


@inherit_docstrings
class RFClassifier(Classifier):
    """
    A Random Forest Classifier
    """
    classifier: RandomForestClassifier

    def __init__(self, n_estimators: int = 10, max_depth: int = None, seed: int = SEED, verbose: int = 0,
                 n_jobs: int = multiprocessing.cpu_count() - 1) -> None:
        """
        Instantiates a RandomForest Classifier with the given parameters
        /!\ Will be overridden if Load=True /!\
        :param n_estimators: The number of trees to build
        :param max_depth: The maximum depth of the trees
        :param seed: The seed to use for the random operations
        :param verbose: The level of logging
        :param n_jobs: The number of threads to use to build the forest
        """
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
                                                 verbose=verbose, n_jobs=n_jobs)

    def get_classifier_name(self) -> str:
        return "RFClassifier"

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
