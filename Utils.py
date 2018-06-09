import os
import sys
import typing
from inspect import getmembers, isfunction

import librosa
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Settings import *

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


def clamp(arr: np.ndarray, l_limit: int = 0, u_limit: int = 1) -> np.ndarray:
    """
    Clamp values in an array to u_limit if value > (u_limit-l_limit)/2 else to l_limit
    :param arr: the array to clamp
    :param l_limit: lower limit
    :param u_limit: upper limit
    :return:
    """
    return np.fromiter(map(lambda value: u_limit if value > (u_limit - l_limit) / 2 else l_limit, arr), count=len(arr),
                       dtype=int)


def file_to_features_with_labels(filename: str) -> typing.Any:
    """
    Extract features and label from an audio file
    :param filename: The filename
    :return: A feat_label_tuple (features, label)
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
    Return an accuracy given predictions and truth values
    :param predictions: The predictions
    :param labels: The true labels
    :return: the accuracy
    """
    assert len(predictions) == len(labels)
    # length = len(predictions)
    # return len(list(filter(lambda t: t[0] == t[1], zip(predictions, labels)))) / float(length)
    return accuracy_score(labels, predictions)


def flatten(arr: np.ndarray) -> np.ndarray:  # np.flatten doesnt seem to work
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


def cut_file(file_tuple: typing.Tuple) -> np.ndarray:
    """
    Cuts a file into smaller windows of equal size
    The window size is equal to the length of the smallest file
    :param file_tuple: The tuple (file_features, label)
    :return: An array of (filecut_features, label) tuples
    """
    features = file_tuple[0]
    label = file_tuple[1]
    new_samples = []
    while len(features) > MIN_SAMPLES_PER_FILE:
        cut_sample = features[:MIN_SAMPLES_PER_FILE]
        features = features[MIN_SAMPLES_PER_FILE:]
        new_samples.append((cut_sample, label))
    if len(features > 0):
        padded_sample = np.pad(features, pad_width=((0, MIN_SAMPLES_PER_FILE - features.shape[0]), (0, 0)),
                               mode='constant')
        new_samples.append((padded_sample, label))
    return np.asarray(new_samples)


def extract_features(features_with_label: np.ndarray) -> np.ndarray:
    """
    Extracts the features from an array of (features, label) tuples
    :param features_with_label: The array
    :return: An array of features
    """
    return np.asarray(list(map(lambda feat_label_tuple: feat_label_tuple[0], features_with_label)))


def extract_labels(features_with_label: np.ndarray) -> np.ndarray:
    """
    Extracts the labels from an array of (features, label) tuples
    :param features_with_label: The array
    :return: An array of labels
    """
    return np.asarray(list(map(lambda feat_label_tuple: feat_label_tuple[1], features_with_label)))


def files_to_features_with_labels(filenames: np.ndarray, one_d: bool = False, split_files=True) -> typing.Any:
    """
    Extracts features and labels from a list of files
    :param filenames: The filenames to use
    :param one_d: If the array is to be flattened or not
    :param split_files: If the features_with_label set is to be split in train/test.
    The test will be unaltered (no padding, flattening, etc) to preserve the concept of file
    :return: Either an array[(cut1_1, label1), (cut1_2, label1), ...] where a cut is of shape (min_samples_per_file, num_features)
    or an array[(sample1_1, label1), (sample1_2, label1), ... (samplen_k, labeln)] where a sample is of shape (1, num_features)
    along with None if split_files=False or an array[(features1, label1), ...] where features is of shape (samples_for_file, num_features)
    """
    if os.path.isfile(FEATURES_WITH_LABEL_FILE):
        features_with_label = load_nparray(FEATURES_WITH_LABEL_FILE)
    else:
        features_with_label = np.asarray([file_to_features_with_labels(file) for file in filenames])
        if not os.path.isfile(MIN_FEATURES_FILE) or not os.path.isfile(MAX_FEATURES_FILE):
            flattened_features = flatten(np.asarray(list(map(lambda t: t[0], features_with_label))))
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

    if split_files:
        features_with_label, test = train_test_split(features_with_label, random_state=SEED, train_size=TRAIN_PERCENT)
    else:
        test = None
    if not one_d:
        two_d_features_with_label = np.asarray(
            list(map(lambda sample: cut_file(sample), features_with_label)))
        two_d_features_with_label = flatten(two_d_features_with_label)

        # Reshape to add the channel dimension
        two_d_features_with_label = np.asarray(list(
            map(lambda sample: (sample[0].reshape(sample[0].shape[0], sample[0].shape[1], 1), sample[1]),
                two_d_features_with_label)))

        return two_d_features_with_label, test
    else:
        one_d_features_with_label = flatten(np.asarray(list(
            map(lambda t: np.asarray(list(map(lambda file_features: (file_features, t[1]), t[0]))),
                features_with_label))))
        return one_d_features_with_label, test


def list_files(dir_name: str, ext=AUDIO_EXT) -> np.ndarray:
    """
    Lists the files in a directory recursively for a given extension
    :param dir_name: The directory to search
    :param ext: The extension of the files to search for
    :return: The array of filenames
    """
    return np.asarray(list(map(lambda path: path.replace("\\", PATH_SEPARATOR),
                               filter(lambda path: path.endswith(ext),
                                      [os.path.join(dp, f) for dp, dn, fn in os.walk(dir_name) for f in fn]))))


def create_gender_file() -> None:
    """
    Creates the Genders.txt file
    :return: None
    """

    def id_gender_tuple(line: str) -> (str, int):
        """
        Creates the (SpeakerID, Gender) feat_label_tuple given a line
        :param line: The line to parse
        :return: The feat_label_tuple
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


def get_genders_dict() -> typing.Dict[str, int]:
    """
    Returns and potentially creates the dictionary (SpeakerID->Gender)
    :return: The dictionary
    """
    global gender_dict
    if gender_dict is None:
        def id_gender_tuple(line: str) -> (str, int):
            """
            Creates the (SpeakerID, Gender) feat_label_tuple given a line
            :param line: The line to parse
            :return: The feat_label_tuple
            """
            t = line.split(GENDERS_FILE_SEPARATOR)
            speaker_id = t[0]
            gender = int(t[1])
            return speaker_id, gender

        fd = open(GENDERS_FILE, "r")
        gender_dict = dict(map(lambda line: id_gender_tuple(line), [line.rstrip() for line in fd]))
    return gender_dict


def inherit_docstrings(cls):
    for name, func in getmembers(cls, isfunction):
        if func.__doc__: continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
    return cls
