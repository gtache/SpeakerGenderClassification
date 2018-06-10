from abc import abstractmethod

import keras as ks
import numpy as np
from sklearn.model_selection import StratifiedKFold

from Settings import SEED
from Utils import inherit_docstrings, get_accuracy
from classifier.Classifier import Classifier


@inherit_docstrings
class NNClassifier(Classifier):
    """
    A base class for a Neural Network classifier.
    """

    model: ks.models.Sequential = None

    @abstractmethod
    def get_model(self) -> ks.models.Sequential:
        """
        :return: the current NN model, and creates it if needed.
        """
        pass

    def cross_validate(self, cv: int, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        skf = StratifiedKFold(n_splits=cv, random_state=SEED)
        scores = []
        for i, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
            self.model = None
            self.model = self.get_model()
            self.train(features[train_idx], labels[train_idx])
            predictions = self.predict(features[test_idx])
            scores.append(get_accuracy(predictions, labels[test_idx]))
            print("Finished split " + str(i))
        scores = np.asarray(scores)
        return scores

    def reset(self) -> None:
        self.model = None
        self.model = self.get_model()
