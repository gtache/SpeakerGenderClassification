import sys
import os
from Utils import *
from classifier.CNNClassifier import CNNClassifier

from classifier.ConstantClassifier import ConstantClassifier
from classifier.RFClassifier import RFClassifier
from classifier.SNNClassifier import SNNClassifier
from classifier.LinearClassifier import LinearClassifier
import numpy as np

SAVE = True
LOAD = False


def main(args=None):
    if args is None:
        args = ["cnn"]
    if args[0] == "const":
        classifier = ConstantClassifier()  # ConstantClassifier
        one_d = True
        train_set, test_set = files_to_features_with_labels(list_files(AUDIO_FILES_DIR), one_d=True)
    elif args[0] == "f":
        classifier = RFClassifier(n_estimators=5, verbose=3)  # RandomForest
        one_d = True
        train_set, test_set = files_to_features_with_labels(list_files(AUDIO_FILES_DIR), one_d=True)
    elif args[0] == "n":
        classifier = SNNClassifier(batch_size=128, num_epochs=300)  # Shallow NN
        one_d = True
        train_set, test_set = files_to_features_with_labels(list_files(AUDIO_FILES_DIR), one_d=True)
    elif args[0] == "svc":
        classifier = LinearClassifier(verbose=3)
        one_d = True
        train_set, test_set = files_to_features_with_labels(list_files(AUDIO_FILES_DIR), one_d=True)
    else:
        classifier = CNNClassifier(batch_size=128, num_epochs=300)  # CNN
        one_d = False
        train_set, test_set = files_to_features_with_labels(list_files(AUDIO_FILES_DIR), one_d=False)
    print("Finished loading/creating features")
    features_train = np.asarray(list(map(lambda t: t[0], train_set)))
    labels_train = np.asarray(list(map(lambda t: t[1], train_set)))
    if not (LOAD and classifier.load(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)):
        classifier.train(features_train, labels_train)
        if SAVE:
            if not os.path.isdir(MODELS_DIR):
                os.mkdir(MODELS_DIR)
            classifier.save(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)

    predictions = []
    labels_test = np.asarray(list(map(lambda tuple: tuple[1], test_set)))
    for tuple in test_set:
        features = tuple[0]
        if one_d:
            results = classifier.predict(features)
            predictions.append(1 if np.sum(results) > len(features) / 2 else 0)
        else:
            features = np.asarray(list(map(lambda t: t[0], cut_file(tuple))))
            features = np.asarray(
                list(map(lambda sample: sample.reshape(sample.shape[0], sample.shape[1], 1), features)))
            results = classifier.predict(features)
            predictions.append(1 if np.sum(results) > len(features) / 2 else 0)
    predictions = np.asarray(predictions)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    with open(OUTPUT_DIR + classifier.get_classifier_name() + "_output.txt", "w") as output_file:
        output_file.writelines([str(pred) + "\n" for pred in predictions])
    print("Test accuracy : "+str(get_accuracy(predictions, labels_test)))


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["cnn"])
