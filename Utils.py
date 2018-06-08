import os
from Settings import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
import numpy as np
import typing
import sys

gender_dict = None
min_shape = sys.maxsize


def audio_to_features(filename: str, n_features: int = FEATURES_NUMBER) -> np.ndarray:
    """
    Extract MFCC features from audio file
    :param filename: The name of the file
    :param n_features: The number of features to extract
    :return: An ndarray of features
    """
    data, samplerate = librosa.load(filename, sr=None)
    mfcc_features = np.asarray(librosa.feature.mfcc(data, samplerate, n_mfcc=n_features))
    global min_shape
    if mfcc_features.shape[1] < min_shape:  # Keep track of min_shape for 2D input
        min_shape = mfcc_features.shape[1]
    return mfcc_features.transpose()


def save_nparray(arr: np.ndarray, filename: str) -> None:
    """
    Save a numpy array to a file
    :param arr: The array to save
    :param filename: The filename to use
    :return: None
    """
    np.save(filename, arr)


def load_nparray(filename: str) -> np.ndarray:
    """
    Load a numpy array given a filename
    :param filename: The filename
    :return: The array
    """
    return np.load(filename)


def audios_to_features(files: typing.Iterable[str]) -> np.ndarray:
    """
    Extract audio features from a collection of files
    :param files: The files
    :return: An array of array of features
    """
    return np.asarray([audio_to_features(file) for file in files])


def split_train_test(features: np.ndarray, labels: np.ndarray, train_size=TRAIN_PERCENT) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Split the dataset into train and test data
    :param features: The features to split
    :param labels: The labels to split
    :param train_size: The size (percentage) of the train set
    :return: A tuple of four array (x_train, x_test, y_train, y_test)
    """
    return train_test_split(features, labels, train_size=train_size, random_state=SEED)


def clamp(arr: np.ndarray) -> np.ndarray:
    return np.fromiter(map(lambda value: 1 if value > 0.5 else 0, arr), count=len(arr), dtype=int)

def file_to_features_with_labels(filename: str, one_d: bool = False) -> typing.Any:
    """
    Extract features and label from an audio file
    :param filename: The filename
    :param one_d: If the data is to be returned on one dimension (flattened)
    :return: Either an array[(sample1,label), (sample2,label) ... (samplen, label)] or a tuple (features, label)
    """
    file_path_split = filename.split(PATH_SEPARATOR)
    id = file_path_split[len(file_path_split) - 1].split(FILE_ID_SEPARATOR)[0].strip()
    global gender_dict
    if gender_dict is None:
        get_genders_dict()
    label = gender_dict[id]
    features = audio_to_features(filename)
    if one_d:
        return np.asarray(list(map(lambda sample: (sample, label), features)))
    else:
        return features, label


def get_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Return an accuracy given predictions and truth values
    :param predictions: The predictions
    :param labels: The true labels
    :return: the accuracy
    """
    assert len(predictions) == len(labels)
    # length = len(predictions)
    # return len(list(filter(lambda t: t[0] == t[1], zip(predictions, labels)))) / float(length)
    return accuracy_score(labels, predictions)


def flatten_features(arr: np.ndarray) -> np.ndarray:  # np.flatten doesnt seem to work
    """
    Flattens an array containing different dimensions
    :param arr: The array to flatten
    :return: The flattened array
    """
    flattened = []
    for subarr in arr:
        for subsubarr in subarr:
            flattened.append(subsubarr)
    return np.asarray(flattened)


def files_to_features_with_labels(filenames: np.ndarray, one_d: bool = False) -> np.ndarray:
    """
    Extracts features and labels from a list of files
    :param filenames: The filenames to use
    :param one_d: If the array is to be flattened or not
    :return: Either an array[(features1, label1), (features2, label2) ...]
    or an array[(sample1_1, label1), (sample1_2, label1), ... (samplen_k, labeln)]
    """
    if (not one_d and os.path.isfile(TWO_D_FEATURES_WITH_LABELS_FILE)) or (
            one_d and os.path.isfile(ONE_D_FEATURES_WITH_LABELS_FILE)):
        return load_nparray(
            ONE_D_FEATURES_WITH_LABELS_FILE if one_d else TWO_D_FEATURES_WITH_LABELS_FILE)
    else:
        if not one_d:
            global min_shape

            def cut_sample(sample: typing.Tuple) -> np.ndarray:
                features = sample[0]
                label = sample[1]
                new_samples = []
                while len(features) > min_shape:
                    cut_sample = features[:min_shape]
                    features = features[min_shape:]
                    new_samples.append((cut_sample, label))
                if len(features > 0):
                    padded_sample = np.pad(features, pad_width=((0, min_shape - features.shape[0]), (0, 0)),
                                           mode='constant')
                    new_samples.append((padded_sample, label))
                return np.asarray(new_samples)

            two_d_features_with_label = np.asarray(
                [file_to_features_with_labels(file, one_d=False) for file in filenames])

            two_d_features_with_label = np.asarray(
                list(map(lambda sample: cut_sample(sample), two_d_features_with_label)))
            two_d_features_with_label = flatten_features(two_d_features_with_label)

            # Reshape to add the channel dimension
            two_d_features_with_label = np.asarray(list(
                map(lambda sample: (sample[0].reshape(sample[0].shape[0], sample[0].shape[1], 1), sample[1]),
                    two_d_features_with_label)))

            save_nparray(two_d_features_with_label, TWO_D_FEATURES_WITH_LABELS_FILE)
            return two_d_features_with_label
        else:
            one_d_features_with_label = np.asarray(
                [file_to_features_with_labels(file, one_d=True) for file in filenames])
            one_d_features_with_label = flatten_features(one_d_features_with_label)
            save_nparray(one_d_features_with_label, ONE_D_FEATURES_WITH_LABELS_FILE)
            return one_d_features_with_label

    # Technically works with only one mfcc extraction but just to be sure, do it more simply
    # Has no impact after generating the files anyway
    #
    # features_with_label = np.asarray([file_to_features_with_labels(file) for file in filenames])
    # global min_shape
    # one_d_features_with_label = flatten_features(np.asarray(
    #     list(map(lambda t: (list(map(lambda f: (f, t[1]), t[0]))), features_with_label))))
    # features_with_label = np.asarray(list(
    #     map(lambda sample: (
    #         np.pad(sample[0], pad_width=((0, 0), (0, min_shape - sample[0].shape[1])), mode='constant'), sample[1]),
    #         features_with_label)))


def list_files(dir: str, ext=AUDIO_EXT) -> np.ndarray:
    """
    Lists the files in a directory recursively for a given extension
    :param dir: The directory to search
    :return: The array of filenames
    """
    return np.asarray(list(map(lambda path: path.replace("\\", PATH_SEPARATOR),
                               filter(lambda path: path.endswith(ext),
                                      [os.path.join(dp, f) for dp, dn, fn in os.walk(dir) for f in fn]))))


def create_gender_file() -> None:
    """
    Creates the Genders.txt file
    :return: None
    """

    def id_gender_tuple(line: str) -> (str, int):
        """
        Creates the (SpeakerID, Gender) tuple given a line
        :param line: The line to parse
        :return: The tuple
        """
        if not line.startswith(COMMENT_STARTER):
            infos = line.split(SPEAKERS_FILE_SEPARATOR)
            id = infos[0].strip()
            gender = infos[1].strip()
            gender_label = F_LABEL if gender == SPEAKERS_F_LABEL else M_LABEL
            return id, gender_label
        else:
            return None

    with open(SPEAKERS_FILE, "r") as fd:
        dic = dict(filter(lambda t: t is not None, map(lambda line: id_gender_tuple(line), fd.readlines())))
        with open(GENDERS_FILE, "w") as output:
            output.writelines(map(lambda t: t[0] + GENDERS_FILE_SEPARATOR + str(t[1]) + "\n", dic.items()))


def get_genders_dict() -> typing.Dict[str, int]:
    """
    Returns and potentially creates the dictionary (SpeakerID->Gender)
    :return: The dictionary
    """
    global gender_dict
    if gender_dict is None:
        def id_gender_tuple(line: str) -> (str, int):
            """
            Creates the (SpeakerID, Gender) tuple given a line
            :param line: The line to parse
            :return: The tuple
            """
            t = line.split(GENDERS_FILE_SEPARATOR)
            id = t[0]
            gender = int(t[1])
            return id, gender

        fd = open(GENDERS_FILE, "r")
        gender_dict = dict(map(lambda line: id_gender_tuple(line), [line.rstrip() for line in fd]))
    return gender_dict
