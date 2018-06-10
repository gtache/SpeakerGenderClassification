from typing import List

from sklearn.model_selection import train_test_split

from Utils import *
from classifier.CNNClassifier import CNNClassifier
from classifier.Classifier import Classifier
from classifier.ConstantClassifier import ConstantClassifier
from classifier.LinearClassifier import LinearClassifier
from classifier.RFClassifier import RFClassifier
from classifier.SNNClassifier import SNNClassifier

SAVE = True
LOAD = False


def run_for_classifier(classifier: Classifier, one_d: bool, cv: int = None,
                       train_set: np.ndarray = None,
                       test_set: np.ndarray = None,
                       save: bool = False,
                       load: bool = False) -> None:
    """
    Test a given classifier and print the results.
    :param classifier: The classifier to test
    :param one_d: If the features are to be flattened or not
    :param cv: If >1, will run cross validation(on samples) with the given number of splits on the classifier
    :param train_set: Optionally given to save time if run_for_classifier is called multiple times
    :param test_set: Optionally given to save time if run_for_classifier is called multiple times
    :param save: If the classifier is to be saved to a file
    :param load: If the classifier is to be loaded from a file
    """

    if train_set is None or test_set is None:
        features_with_label = files_to_features_with_labels(list_files(AUDIO_FILES_DIR))
        train_set, test_set = train_test_split(features_with_label, random_state=SEED, train_size=TRAIN_PERCENT,
                                               test_size=1 - TRAIN_PERCENT)
    print("Finished loading/creating features")
    print("Using classifier " + classifier.get_classifier_name())

    # Run cross validation
    if cv is not None and cv > 1:
        print("Running cross validation")
        cv_set = np.append(train_set, test_set, axis=0)
        if one_d:
            cv_set = to_1d(cv_set)
        else:
            cv_set = to_2d(cv_set)
        scores = classifier.cross_validate(CV, extract_features(cv_set), extract_labels(cv_set))
        print("CV Score : Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
        classifier.reset()

    if one_d:
        train_set = to_1d(train_set)
    else:
        train_set = to_2d(train_set)

    features_train = extract_features(train_set)
    labels_train = extract_labels(train_set)

    if not (load and classifier.load(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)):
        print("Training " + classifier.get_classifier_name())
        classifier.train(features_train, labels_train)
        if save:
            if not os.path.isdir(MODELS_DIR):
                os.mkdir(MODELS_DIR)
            classifier.save(MODELS_DIR + classifier.get_classifier_name() + DUMP_EXT)
            print("Saved " + classifier.get_classifier_name())
    else:
        print("Loaded " + classifier.get_classifier_name())

    # Per file predictions
    print("Predicting on files...")
    predictions = []
    test_labels = extract_labels(test_set)
    for feat_label_tuple in test_set:
        features = feat_label_tuple[0]
        if not one_d:
            features = extract_features(cut_file(feat_label_tuple))
            # Add depth dimension
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
        transformed_test_set = to_1d(test_set)
    else:
        transformed_test_set = to_2d(test_set)

    samples_features = extract_features(transformed_test_set)
    samples_predictions = classifier.predict(samples_features)
    samples_test_labels = extract_labels(transformed_test_set)

    print("Test accuracy - files : " + str(get_accuracy(predictions, test_labels)))
    print("Test accuracy - samples : " + str(get_accuracy(samples_predictions, samples_test_labels)))


def main(args: List[str] = None):
    """
    Main function of the program.
    :param args: The optional arguments
    """
    one_d = True

    if args is None:
        args = ["cnn"]
    if args[0] == "const":
        classifier = ConstantClassifier()  # ConstantClassifier
    elif args[0] == "f":
        classifier = RFClassifier(n_estimators=5, verbose=1)  # RandomForest
    elif args[0] == "n":
        classifier = SNNClassifier(batch_size=128, num_epochs=300, verbose=1)  # Shallow Neural Net
    elif args[0] == "svc":
        classifier = LinearClassifier(c=1, verbose=1)  # Linear SVM
    else:
        classifier = CNNClassifier(batch_size=128, num_epochs=300, verbose=1)  # Convolutional Neural Net
        one_d = False
    run_for_classifier(classifier, one_d, save=SAVE, load=LOAD)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["cnn"])
