from Utils import *
from classifier.CNNClassifier import CNNClassifier
from classifier.Classifier import Classifier
from classifier.ConstantClassifier import ConstantClassifier
from classifier.LinearClassifier import LinearClassifier
from classifier.RFClassifier import RFClassifier
from classifier.SNNClassifier import SNNClassifier

SAVE = True
LOAD = False


def run_for_classifier(classifier: Classifier, one_d: bool,
                       train_set: np.ndarray = None,
                       test_set: np.ndarray = None) -> None:
    """
    Test a given classifier and print the results
    :param classifier: The classifier to test
    :param one_d: If the features are to be fed flattened or not
    :param train_set: Optionally given to save time if run_for_classifier is called multiple times
    :param test_set: Optionally given to save time if run_for_classifier is called multiple times
    """
    if train_set is None or test_set is None:
        train_set, test_set = files_to_features_with_labels(list_files(AUDIO_FILES_DIR), one_d=one_d, split_files=True)
    print("Finished loading/creating features")
    features_train = extract_features(train_set)
    labels_train = extract_labels(train_set)
    if not (LOAD and classifier.load(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)):
        print("Training " + classifier.get_classifier_name())
        classifier.train(features_train, labels_train)
        if SAVE:
            if not os.path.isdir(MODELS_DIR):
                os.mkdir(MODELS_DIR)
            classifier.save(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)
            print("Saved " + classifier.get_classifier_name())
    else:
        print("Loaded " + classifier.get_classifier_name())

    def return_majority(arr: np.ndarray) -> int:
        """
        Return the majority label (0 or 1) in an array
        :param arr: The array
        :return: 0 or 1
        """
        return 1 if np.sum(arr) > len(arr) / 2 else 0

    # Per file predictions
    print("Predicting on files...")
    predictions = []
    test_labels = extract_labels(test_set)
    for feat_label_tuple in test_set:
        features = feat_label_tuple[0]
        if one_d:
            results = classifier.predict(features)
            predictions.append(return_majority(results))
        else:
            features = extract_features(cut_file(feat_label_tuple))
            features = np.asarray(
                list(map(lambda sample: sample.reshape(sample.shape[0], sample.shape[1], 1), features)))
            results = classifier.predict(features)
            predictions.append(return_majority(results))
    predictions = np.asarray(predictions)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    with open(OUTPUT_DIR + classifier.get_classifier_name() + "_output.txt", "w") as output_file:
        output_file.writelines([str(pred) + "\n" for pred in predictions])

    # Per sample predictions
    print("Predicting on samples...")
    if one_d:
        flattened_test_set = flatten(np.asarray(list(map(lambda file_features: np.asarray(
            list(map(lambda sample_features: (sample_features, file_features[1]), file_features[0]))), test_set))))
        samples_features = extract_features(flattened_test_set)
        samples_predictions = classifier.predict(samples_features)
        samples_test_labels = extract_labels(flattened_test_set)
    else:
        cut_test_set = flatten(np.asarray(list(map(lambda file_features: cut_file(file_features), test_set))))
        cut_features = extract_features(cut_test_set)
        cut_features = np.asarray(
            list(map(lambda sample: sample.reshape(sample.shape[0], sample.shape[1], 1), cut_features)))
        samples_predictions = classifier.predict(cut_features)
        samples_test_labels = extract_labels(cut_test_set)

    print("Test accuracy - files : " + str(get_accuracy(predictions, test_labels)))
    print("Test accuracy - samples : " + str(get_accuracy(samples_predictions, samples_test_labels)))


def main(args=None):
    """
    Main function of the program
    :param args: The optional arguments
    """
    one_d = True

    if args is None:
        args = ["cnn"]
    if args[0] == "const":
        classifier = ConstantClassifier()  # ConstantClassifier
    elif args[0] == "f":
        classifier = RFClassifier(n_estimators=200, verbose=0)  # RandomForest
    elif args[0] == "n":
        classifier = SNNClassifier(batch_size=128, num_epochs=200)  # Shallow Neural Net
    elif args[0] == "svc":
        classifier = LinearClassifier(c=1, verbose=1)  # Linear SVM
    else:
        classifier = CNNClassifier(batch_size=128, num_epochs=200)  # Convolutional Neural Net
        one_d = False
    run_for_classifier(classifier, one_d)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["cnn"])
