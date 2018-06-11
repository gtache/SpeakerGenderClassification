import numpy as np

from Settings import *
from classifier.NNClassifier import NNClassifier

np.random.seed(seed=SEED)

import keras as ks
from keras.layers import *
from Utils import inherit_docstrings


@inherit_docstrings
class SNNClassifier(NNClassifier):
    """
    A Shallow Neural Net with a single hidden layer.
    """
    model: ks.models.Sequential = None

    def __init__(self, validation_percentage: float = VALIDATION_PERCENT, batch_size: int = BATCH_SIZE,
                 num_epochs: int = NUM_EPOCHS, learning_rate: float = LEARNING_RATE, input_dim: int = FEATURES_NUMBER,
                 num_units: int = 256,
                 verbose: int = 0):
        """
        Instantiate a SNN with the given parameters.
        Will be overridden if load is called later.
        :param validation_percentage: The percentage of samples to use for validation
        :param batch_size: The batch size
        :param num_epochs: The maximum number of epochs to allow for training
        :param learning_rate: The base learning rate
        :param input_dim: The input dimension of the input layer (i.e. the number of features per sample)
        :param num_units: The number of units for the hidden Dense layer
        :param verbose: The level of logging
        """
        super().__init__(validation_percentage=validation_percentage, batch_size=batch_size, num_epochs=num_epochs,
                         learning_rate=learning_rate, verbose=verbose)
        self.input_dim = input_dim
        self.num_units = num_units

    def get_model(self) -> ks.models.Sequential:
        if self.model is None:
            self.model = ks.Sequential()
            self.model.add(
                Dense(self.num_units, use_bias=True, input_dim=self.input_dim,
                      kernel_initializer=ks.initializers.glorot_normal(seed=SEED)))
            self.model.add(PReLU())
            # self.model.add(Dropout(rate=0.3, seed=SEED))
            self.model.add(Dense(1, use_bias=True, kernel_initializer=ks.initializers.glorot_normal(seed=SEED)))
            self.model.add(Activation("sigmoid"))
            print(self.model.summary())
        return self.model

    def get_classifier_name(self) -> str:
        return "SNNClassifier - units " + str(self.num_units)
