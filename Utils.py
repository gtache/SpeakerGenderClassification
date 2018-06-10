import os
import sys
from inspect import getmembers, isfunction
from typing import Iterable, Tuple, Any, Dict

import librosa
import numpy as np
from sklearn.metrics import accuracy_score

from Settings import *

gender_dict: Dict[str, int] = None
min_shape: int = sys.maxsize


def audio_to_features(filename: str, n_features: int = FEATURES_NUMBER) -> np.ndarray:
    """
    Extract MFCC features from audio file using librosa.
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
    Save a numpy array to a file.
    :param arr: The array to save
    :param filename: The filename to use
    """
    np.save(filename, arr)


def load_nparray(filename: str) -> np.ndarray:
    """
    Load a numpy array given a filename.
    :param filename: The filename
    :return: The array
    """
    return np.load(filename)


def audios_to_features(files: Iterable[str]) -> np.ndarray:
    """
    Extract audio features from a collection of files.
    :param files: The files
    :return: An array of array of features
    """
    return np.asarray([audio_to_features(file) for file in files])


def clamp(arr: np.ndarray, lower_value: int = 0, upper_value: int = 1) -> np.ndarray:
    """
    Clamp values in an array to upper_value if value > (upper_value-lower_value)/2 else to lower_value.
    :param arr: the array to clamp
    :param lower_value: lower limit
    :param upper_value: upper limit
    :return: The clamped array
    """
    return np.fromiter(map(lambda value: upper_value if value > (upper_value - lower_value) / 2 else lower_value, arr),
                       count=len(arr),
                       dtype=int)


def file_to_features_with_labels(filename: str) -> Any:
    """
    Extract features and label from an audio file.
    :param filename: The filename
    :return: A tuple (features, label)
    """
    file_path_split = filename.split(PATH_SEPARATOR)
    speaker_id = file_path_split[len(file_path_split) - 1].split(FILE_ID_SEPARATOR)[0].strip()
    global gender_dict
    if gender_dict is None:
        get_genders_dict()
    label = gender_dict[speaker_id]
    features = audio_to_features(filename)
    return features, label


def get_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Return an accuracy given predictions and true labels.
    :param predictions: The predictions
    :param labels: The true labels
    :return: the accuracy
    """
    assert len(predictions) == len(labels)
    return accuracy_score(labels, predictions)


def flatten(arr: Iterable) -> np.ndarray:  # np.flatten doesnt seem to work
    """
    Flatten an array (two-level deeps, not recursively) containing different dimensions.
    :param arr: The array to flatten
    :return: The flattened array
    """
    flattened = []
    for subarr in arr:
        for subsubarr in subarr:
            flattened.append(subsubarr)
    return np.asarray(flattened)


def cut_file(file_tuple: Tuple) -> np.ndarray:
    """
    Cut a file into smaller windows of equal size.
    The window size is equal to the length of the smallest file.
    :param file_tuple: The tuple (file_features, label)
    :return: An array of (filecut_features, label) tuples
    """
    features = file_tuple[0]
    label = file_tuple[1]
    new_samples = []
    while len(features) > FEATURES_WINDOW_SIZE:
        cut_sample = features[:FEATURES_WINDOW_SIZE]
        features = features[FEATURES_WINDOW_SIZE:]
        new_samples.append((cut_sample, label))
    if len(features > 0):
        padded_sample = np.pad(features, pad_width=((0, FEATURES_WINDOW_SIZE - features.shape[0]), (0, 0)),
                               mode='constant')
        new_samples.append((padded_sample, label))
    return np.asarray(new_samples)


def extract_features(features_with_label: Iterable) -> np.ndarray:
    """
    Extract the features from an array of (features, label) tuples.
    :param features_with_label: The array
    :return: An array of features
    """
    return np.asarray(list(map(lambda feat_label_tuple: feat_label_tuple[0], features_with_label)))


def extract_labels(features_with_label: Iterable) -> np.ndarray:
    """
    Extract the labels from an array of (features, label) tuples.
    :param features_with_label: The array
    :return: An array of labels
    """
    return np.asarray(list(map(lambda feat_label_tuple: feat_label_tuple[1], features_with_label)))


def to_2d(features_with_label: np.ndarray) -> np.ndarray:
    """
    Prepare the features to be fed to as a 2D input.
    :param features_with_label: The features along with their label
    :return: The transformed features along with their label
    """
    two_d_features_with_label = flatten(list(map(lambda sample: cut_file(sample), features_with_label)))

    # Reshape to add the channel dimension
    return np.asarray(list(
        map(lambda sample: (sample[0].reshape(sample[0].shape[0], sample[0].shape[1], 1), sample[1]),
            two_d_features_with_label)))


def to_1d(features_with_label: np.ndarray) -> np.ndarray:
    """
    Prepare the features to be fed to as a 1D input.
    :param features_with_label: The features along with their label
    :return: The transformed features along with their label
    """
    return flatten(list(
        map(lambda t: np.asarray(list(map(lambda file_features: (file_features, t[1]), t[0]))),
            features_with_label)))


def files_to_features_with_labels(filenames: Iterable[str]) -> np.ndarray:
    """
    Extract features and labels from a list of files.
    :param filenames: The filenames to use
    :return: An array of (features, label) tuples
    """
    if os.path.isfile(FEATURES_WITH_LABEL_FILE):
        features_with_label = load_nparray(FEATURES_WITH_LABEL_FILE)
    else:
        features_with_label = np.asarray([file_to_features_with_labels(file) for file in filenames])
        if not os.path.isfile(MIN_FEATURES_FILE) or not os.path.isfile(MAX_FEATURES_FILE):
            flattened_features = flatten(extract_features(features_with_label))
            min_f = flattened_features.min(axis=0)
            save_nparray(min_f, MIN_FEATURES_FILE)
            max_f = flattened_features.max(axis=0)
            save_nparray(max_f, MAX_FEATURES_FILE)
        else:
            min_f = load_nparray(MIN_FEATURES_FILE)
            max_f = load_nparray(MAX_FEATURES_FILE)

        # Normalize the features
        features_with_label = np.asarray(list(
            map(lambda feat_label_tuple: (
                np.asarray(list(map(lambda sample: (sample - min_f) / (max_f - min_f), feat_label_tuple[0]))),
                feat_label_tuple[1]),
                features_with_label)))

        save_nparray(features_with_label, FEATURES_WITH_LABEL_FILE)

    return features_with_label


def list_files(dir_name: str, ext=AUDIO_EXT) -> np.ndarray:
    """
    List the files in a directory recursively for a given extension.
    :param dir_name: The directory to search
    :param ext: The extension of the files to search for
    :return: The array of filenames
    """
    return np.asarray(list(map(lambda path: path.replace("\\", PATH_SEPARATOR),
                               filter(lambda path: path.endswith(ext),
                                      [os.path.join(dp, f) for dp, dn, fn in os.walk(dir_name) for f in fn]))))


def create_gender_file() -> None:
    """
    Create the Genders.txt file.
    """

    def id_gender_tuple(line: str) -> (str, int):
        """
        Creates the (SpeakerID, Gender) tuple given a line.
        :param line: The line to parse
        :return: The tuple
        """
        if not line.startswith(COMMENT_STARTER):
            infos = line.split(SPEAKERS_FILE_SEPARATOR)
            speaker_id = infos[0].strip()
            gender = infos[1].strip()
            gender_label = F_LABEL if gender == SPEAKERS_F_LABEL else M_LABEL
            return speaker_id, gender_label
        else:
            return None

    with open(SPEAKERS_FILE, "r") as fd:
        dic = dict(filter(lambda t: t is not None, map(lambda line: id_gender_tuple(line), fd.readlines())))
    with open(GENDERS_FILE, "w") as output:
        output.writelines(map(lambda t: t[0] + GENDERS_FILE_SEPARATOR + str(t[1]) + "\n", dic.items()))


def get_genders_dict() -> Dict[str, int]:
    """
    Return and potentially creates the dictionary (SpeakerID->Gender).
    :return: The dictionary
    """
    global gender_dict
    if gender_dict is None:
        def id_gender_tuple(line: str) -> (str, int):
            """
            Create the (SpeakerID, Gender) tuple given a line.
            :param line: The line to parse
            :return: The tuple
            """
            t = line.split(GENDERS_FILE_SEPARATOR)
            speaker_id = t[0]
            gender = int(t[1])
            return speaker_id, gender

        if not os.path.isfile(GENDERS_FILE):
            create_gender_file()
        fd = open(GENDERS_FILE, "r")
        gender_dict = dict(map(lambda line: id_gender_tuple(line), [line.rstrip() for line in fd]))
    return gender_dict


def return_majority(arr: np.ndarray) -> int:
    """
    Return the majority label (0 or 1) in an array.
    :param arr: The array
    :return: 0 or 1
    """
    return 1 if np.sum(arr) > len(arr) / 2 else 0


def inherit_docstrings(cls):
    """
    Used to make a class inherit methods docstrings from its parent.
    :param cls: The class
    :return: the class
    """
    for name, func in getmembers(cls, isfunction):
        if func.__doc__:
            continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
    return cls
