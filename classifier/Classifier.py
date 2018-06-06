from abc import ABC, abstractmethod
import numpy as np
import os


class Classifier(ABC):
    """
    Base class for a classifier
    """

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray, cv: int = None) -> None:
        pass

    @abstractmethod
    def save(self, filename: str) -> None:
        pass

    @abstractmethod
    def load(self, filename: str) -> None:
        pass

    @abstractmethod
    def get_classifier_name(self) -> str:
        pass

    @staticmethod
    def check_dump_exists(filename) -> bool:
        return os.path.isfile(filename)
