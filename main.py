import sys
import os
from Utils import *
from classifier.CNNClassifier import CNNClassifier

from classifier.ConstantClassifier import ConstantClassifier
from classifier.RFClassifier import RFClassifier
from classifier.SNNClassifier import SNNClassifier
import numpy as np

SAVE = True
LOAD = False


def main(args=None):
    if args is None:
        args = ["cnn"]
    if args[0] == "const":
        classifier = ConstantClassifier()  # ConstantClassifier
        features_with_label = files_to_features_with_labels(list_audio_files(AUDIO_FILES_DIR), one_d=True)
    elif args[0] == "f":
        classifier = RFClassifier(n_estimators=100, verbose=3)  # RandomForest
        features_with_label = files_to_features_with_labels(list_audio_files(AUDIO_FILES_DIR), one_d=True)
    elif args[0] == "n":
        classifier = SNNClassifier()  # Shallow NN
        features_with_label = files_to_features_with_labels(list_audio_files(AUDIO_FILES_DIR), one_d=True)
    else:
        classifier = CNNClassifier()  # CNN
        features_with_label = files_to_features_with_labels(list_audio_files(AUDIO_FILES_DIR), one_d=False)
    print("Finished loading/creating features")
    features = np.asarray(list(map(lambda t: t[0], features_with_label)))
    labels = np.asarray(list(map(lambda t: t[1], features_with_label)))
    x_train, x_test, y_train, y_test = split_train_test(features, labels)
    if not (LOAD and classifier.load(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)):
        classifier.train(x_train, y_train)
        if SAVE:
            if not os.path.isdir(MODELS_DIR):
                os.mkdir(MODELS_DIR)
            classifier.save(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)

    predictions = classifier.predict(x_test)
    with open(OUTPUT_DIR + classifier.get_classifier_name() + "_output.txt", "w") as output_file:
        output_file.writelines([str(pred) + "\n" for pred in predictions])
    print(get_accuracy(predictions, y_test))


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["cnn"])
