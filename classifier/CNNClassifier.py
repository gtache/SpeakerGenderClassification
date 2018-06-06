from classifier.Classifier import Classifier
import numpy as np
import keras as ks
from keras import backend as K
from keras.layers import *
from Settings import *

class CNNClassifier(Classifier):

    strides = (1, 1)
    model = None

    def __init__(self, validation_percentage=VALIDATION_PERCENT, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                 learning_rate=LEARNING_RATE,
                 input_dim=3, filter_depth=FILTER_DEPTH):
        pass

    def get_classifier_name(self) -> str:
        return "CNNClassifier"

    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    def train(self, features: np.ndarray, labels: np.ndarray, cv: int = None) -> None:
        pass

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> bool:
        pass
