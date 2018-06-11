import os
from abc import abstractmethod

import keras as ks
import numpy as np
from sklearn.model_selection import StratifiedKFold

from Settings import *
from Utils import inherit_docstrings, get_accuracy, clamp
from classifier.Classifier import Classifier


@inherit_docstrings
class NNClassifier(Classifier):
    """
    A base class for a Neural Network classifier.
    """

    model: ks.models.Sequential = None

    @abstractmethod
    def __init__(self, validation_percentage: float = VALIDATION_PERCENT, batch_size: int = BATCH_SIZE,
                 num_epochs: int = NUM_EPOCHS, learning_rate: float = LEARNING_RATE, verbose: int = 0):
        """
        Instantiates a Neural network with the given parameters
        :param validation_percentage: The percentage of samples to use for validation
        :param batch_size: The batch size
        :param num_epochs: The max number of epochs to allow for training
        :param learning_rate: The base learning rate
        :param verbose: The level of logging
        """
        super().__init__(verbose)
        self.validation_percentage = validation_percentage
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

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

    def predict(self, features: np.ndarray) -> np.ndarray:
        return clamp(self.get_model().predict(features, batch_size=self.batch_size))

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        optimizer = ks.optimizers.Adam(lr=self.learning_rate)
        early_stop_callback = ks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=15, verbose=1)
        learning_rate_callback = ks.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)
        if not os.path.isdir(TENSORBOARD_DIR):
            os.mkdir(TENSORBOARD_DIR)
        tensorboard_callback = ks.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=0, write_graph=True,
                                                        write_images=True)

        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)

        # save_callback = ks.callbacks.ModelCheckpoint(MODELS_DIR + "SNN.{epoch:02d}-{val_loss:.2f}.keras", verbose=0)
        callbacks = [early_stop_callback, learning_rate_callback, tensorboard_callback]

        self.get_model().compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.get_model().fit(x=features, y=labels, batch_size=self.batch_size, epochs=self.num_epochs,
                             validation_split=self.validation_percentage, callbacks=callbacks, verbose=self.verbose)

    def save(self, filename: str) -> None:
        self.model.save(filename)

    def load(self, filename: str) -> bool:
        if self.check_dump_exists(filename):
            self.model = ks.models.load_model(filename)
            return True
        else:
            return False
