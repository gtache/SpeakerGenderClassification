from classifier.Classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from Settings import SEED
from sklearn.externals import joblib
import numpy as np
import os


class RFClassifier(Classifier):
    def get_classifier_name(self) -> str:
        return "RFClassifier"

    classifier = None

    def __init__(self, n_estimators: int = 10, max_depth: int = None, seed=SEED, verbose=0, n_jobs=7) -> None:
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
                                                 verbose=verbose, n_jobs=n_jobs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(features)

    def train(self, features: np.ndarray, labels: np.ndarray, cv: int = None) -> None:
        if cv is None or cv < 2:
            self.classifier.fit(features, labels)
        else:
            cross_val_score(self.classifier, features, labels, cv)

    def save(self, filename: str) -> None:
        joblib.dump(self.classifier, filename, compress=5)

    def load(self, filename: str) -> bool:
        if self.check_dump_exists(filename):
            self.classifier = joblib.load(filename)
            return True
        else:
            return False
