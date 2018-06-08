from classifier.Classifier import Classifier
from Settings import *
import numpy as np
import os

np.random.seed(seed=SEED)

import keras as ks
from keras.layers import *
from Utils import clamp


class SNNClassifier(Classifier):
    model = None

    def __init__(self, validation_percentage=VALIDATION_PERCENT, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                 learning_rate=LEARNING_RATE,
                 input_dim=FEATURES_NUMBER):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.validation_percentage = validation_percentage
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_model(self):
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
