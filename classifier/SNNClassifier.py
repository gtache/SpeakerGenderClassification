import os

import numpy as np

from Settings import *
from classifier.Classifier import Classifier

np.random.seed(seed=SEED)

import keras as ks
from keras.layers import *
from Utils import clamp, inherit_docstrings


@inherit_docstrings
class SNNClassifier(Classifier):
    """
    A Shallow Neural Net with a single hidden layer
    """
    model: ks.models.Sequential

    def __init__(self, validation_percentage: float = VALIDATION_PERCENT, batch_size: int = BATCH_SIZE,
                 num_epochs: int = NUM_EPOCHS,
                 learning_rate: float = LEARNING_RATE,
                 input_dim: int = FEATURES_NUMBER):
        """
        Instantiates a SNN with the given parameters
        /!\ Will be overridden if Load=True /!\
        :param validation_percentage: The percentage of samples to use for validation
        :param batch_size: The batch size
        :param num_epochs: The maximum number of epochs to allow for training
        :param learning_rate: The base learning rate
        :param input_dim: The input dimension of the input layer (i.e. the number of features per sample)
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.validation_percentage = validation_percentage
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_model(self) -> ks.models.Sequential:
        """
        :return: The current model, or creates it if needed
        """
        if self.model is None:
            self.model = ks.Sequential()
            self.model.add(
                Dense(200, use_bias=True, input_dim=self.input_dim,
                      kernel_initializer=ks.initializers.glorot_normal(seed=SEED)))
            self.model.add(PReLU())
            # self.model.add(Dropout(rate=0.3, seed=SEED))
            self.model.add(Dense(1, use_bias=True, kernel_initializer=ks.initializers.glorot_normal(seed=SEED)))
            # self.model.add(Dropout(rate=0.3, seed=SEED))
            self.model.add(Activation("sigmoid"))
            print(self.model.summary())
        return self.model

    def get_classifier_name(self) -> str:
        return "SNNClassifier"

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

        self.get_model().fit(x=features, y=labels,
                             batch_size=self.batch_size, epochs=self.num_epochs,
                             validation_split=self.validation_percentage, callbacks=callbacks)

    def save(self, filename: str) -> None:
        self.model.save(filename)

    def load(self, filename: str) -> bool:
        if self.check_dump_exists(filename):
            self.model = ks.models.load_model(filename)
            return True
        else:
            return False
