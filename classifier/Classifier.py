import os
from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """
    Base class for a classifier
    """

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the labels given the features
        :param features: The features to use
        :return: The labels predicted by the classifier
        """
        pass

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the classifier
        :param features: The features to use
        :param labels: The truth labels corresponding to the features
        :return: None
        """
        pass

    @abstractmethod
    def save(self, filename: str) -> None:
        """
        Save the classifier to a file
        :param filename: The filename to save to
        :return: None
        """
        pass

    @abstractmethod
    def load(self, filename: str) -> bool:
        """
        Load the classifier from a file
        :param filename: The filename to load from
        :return: True if it loaded successfully, false otherwise
        """
        pass

    @abstractmethod
    def get_classifier_name(self) -> str:
        """
        :return: the name of the classifier
        """
        pass

    @staticmethod
    def check_dump_exists(filename) -> bool:
        """
        Check if a classifier dump exists for the given filename
        :param filename: The filename to check
        :return: True if it is valid, false otherwise
        """
        return os.path.isfile(filename)
