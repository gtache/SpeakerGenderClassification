import os

import numpy as np

from Settings import *
from classifier.NNClassifier import NNClassifier

np.random.seed(seed=SEED)
import keras as ks
from keras.layers import *
from Utils import clamp, inherit_docstrings
from typing import Tuple


@inherit_docstrings
class CNNClassifier(NNClassifier):
    """
    A deep Convolutional Neural Network classifier.
    """
    model: ks.models.Sequential = None

    def __init__(self, validation_percentage: float = VALIDATION_PERCENT, batch_size: int = BATCH_SIZE,
                 num_epochs: int = NUM_EPOCHS, learning_rate: float = LEARNING_RATE,
                 input_shape: Tuple[int, int, int] = INPUT_SHAPE, filter_depth: int = FILTER_DEPTH,
                 kernel_size: Tuple[int, int] = KERNEL_SIZE, strides: Tuple[int, int] = STRIDES,
                 verbose: int = 0) -> None:
        """
        Instantiate a CNN with the given parameters.
        Will be overridden if load is called later.
        :param validation_percentage: The percentage of samples to use for validation
        :param batch_size: The batch size
        :param num_epochs: The max number of epochs to allow for training
        :param learning_rate: The base learning rate
        :param input_shape: The input shape of the input layer (i.e. (window_size_x, window_size_y, num_channels))
        :param filter_depth: The filter depth for the Conv2D layers
        :param kernel_size: The kernel size for the Conv2D layers
        :param strides: The strides for the Conv2D and MaxPool2D layers
        :param verbose: The level of logging
        """
        super().__init__(verbose)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.validation_percentage = validation_percentage
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_epochs = num_epochs
        self.strides = strides
        self.filter_depth = filter_depth

    def get_model(self) -> ks.models.Sequential:
        if self.model is None:
            self.model = ks.Sequential()
            self.model.add(BatchNormalization(momentum=0.1, input_shape=self.input_shape))
            self.model.add(
                Conv2D(filters=self.filter_depth * 2 ** 0,
                       kernel_size=self.kernel_size,
                       strides=self.strides,
                       padding='valid',
                       use_bias=True,
                       kernel_initializer=ks.initializers.glorot_normal(seed=SEED),
                       kernel_regularizer=ks.regularizers.l2(0.01),
                       input_shape=self.input_shape))
            self.model.add(PReLU())
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(MaxPooling2D(pool_size=(2, 2),
                                        strides=self.strides,
                                        padding='valid'))
            self.model.add(Dropout(0.3, seed=SEED))
            self.model.add(Conv2D(filters=self.filter_depth * 2 ** 1,
                                  kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  padding='valid',
                                  use_bias=True,
                                  kernel_initializer=ks.initializers.glorot_normal(seed=SEED),
                                  kernel_regularizer=ks.regularizers.l2(0.01)
                                  ))
            self.model.add(PReLU())
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(MaxPooling2D(pool_size=(2, 2),
                                        strides=self.strides,
                                        padding='valid'))
            self.model.add(Dropout(0.3, seed=SEED))
            self.model.add(Flatten())
            self.model.add(Dense(512,
                                 use_bias=True,
                                 kernel_initializer=ks.initializers.glorot_normal(seed=SEED),
                                 # activity_regularizer=ks.regularizers.l2(0.01),
                                 kernel_regularizer=ks.regularizers.l2(0.01)))
            self.model.add(PReLU())
            self.model.add(Dropout(0.5, seed=SEED))
            self.model.add(Dense(256,
                                 use_bias=True,
                                 kernel_initializer=ks.initializers.glorot_normal(seed=SEED),
                                 # activity_regularizer=ks.regularizers.l2(0.01),
                                 kernel_regularizer=ks.regularizers.l2(0.01)))
            self.model.add(PReLU())
            self.model.add(Dropout(0.5, seed=SEED))
            self.model.add(Dense(1,
                                 use_bias=True,
                                 kernel_initializer=ks.initializers.glorot_normal(seed=SEED)))
            self.model.add(Activation("sigmoid"))
            print(self.model.summary())
        return self.model

    def get_classifier_name(self) -> str:
        return "CNNClassifier"

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
