import os
from Settings import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
import numpy as np
import typing
from tqdm import tqdm

gender_dict = None

max_shape = 0


def audio_to_features(file: str) -> np.ndarray:
    data, samplerate = librosa.load(file, sr=None)
    mfcc_features = np.asarray(librosa.feature.mfcc(data, samplerate, n_mfcc=FEATURES_NUMBER))
    global max_shape
    if mfcc_features.shape[1] > max_shape:
        max_shape = mfcc_features.shape[1]
    return mfcc_features


def save_features_with_label(features_with_label: np.ndarray, filename: str) -> None:
    np.save(filename, features_with_label)


def load_features_with_label(filename: str) -> np.ndarray:
    return np.load(filename)


def audios_to_features(files: typing.Iterable[str]) -> np.ndarray:
    return np.asarray([audio_to_features(file) for file in files])


def split_train_test(features: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    return train_test_split(features, labels, train_size=TRAIN_PERCENT, random_state=SEED)


def file_to_features_with_labels(file: str, one_d: bool = False) -> typing.Any:
    file_path_split = file.split(PATH_SEPARATOR)
    id = file_path_split[len(file_path_split) - 1].split(FILE_ID_SEPARATOR)[0].strip()
    global gender_dict
    if gender_dict is None:
        get_genders_dict()
    label = gender_dict[id]
    features = audio_to_features(file)
    if one_d:
        return np.asarray(list(map(lambda sample: (sample, label), features.transpose())))
    else:
        return features, label


def get_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    assert len(predictions) == len(labels)
    # length = len(predictions)
    # return len(list(filter(lambda t: t[0] == t[1], zip(predictions, labels)))) / float(length)
    return accuracy_score(labels, predictions)


def flatten_features(arr: np.ndarray) -> np.ndarray:  # np.flatten doesnt seem to work
    flattened = []
    for subarr in arr:
        for subsubarr in subarr:
            flattened.append(subsubarr)
    return np.asarray(flattened)


def files_to_features_with_labels(files: np.ndarray, one_d: bool = False) -> np.ndarray:
    if (not one_d and os.path.isfile(TWO_D_FEATURES_WITH_LABELS_FILE)) or (
            one_d and os.path.isfile(ONE_D_FEATURES_WITH_LABELS_FILE)):
        return load_features_with_label(
            ONE_D_FEATURES_WITH_LABELS_FILE if one_d else TWO_D_FEATURES_WITH_LABELS_FILE)
    else:
        if not one_d:
            two_d_features_with_label = np.asarray([file_to_features_with_labels(file, one_d=False) for file in files])
            global max_shape

            # Pad to get the same number of features per sample, for CNN
            two_d_features_with_label = np.asarray(list(map(lambda sample: (
                np.pad(sample[0], pad_width=((0, 0), (0, max_shape - sample[0].shape[1])), mode='constant'), sample[1]),
                                                            two_d_features_with_label)))

            # Reshape to add the depth dimension

            two_d_features_with_label = np.asarray(list(
                map(lambda sample: (sample[0].reshape(sample[0].shape[0], sample[0].shape[1], 1), sample[1]),
                    two_d_features_with_label)))

            save_features_with_label(two_d_features_with_label, TWO_D_FEATURES_WITH_LABELS_FILE)
            return two_d_features_with_label
        else:
            one_d_features_with_label = np.asarray([file_to_features_with_labels(file, one_d=True) for file in files])
            one_d_features_with_label = flatten_features(one_d_features_with_label)
            save_features_with_label(one_d_features_with_label, ONE_D_FEATURES_WITH_LABELS_FILE)
            return one_d_features_with_label

    # Theoretically works with only one mfcc extraction but just to be sure, do it more simply
    # Has no impact after generating the files anyway
    #
    # features_with_label = np.asarray([file_to_features_with_labels(file) for file in files])
    # global max_shape
    # one_d_features_with_label = flatten_features(np.asarray(
    #     list(map(lambda t: (list(map(lambda f: (f, t[1]), t[0]))), features_with_label))))
    # features_with_label = np.asarray(list(
    #     map(lambda sample: (
    #         np.pad(sample[0], pad_width=((0, 0), (0, max_shape - sample[0].shape[1])), mode='constant'), sample[1]),
    #         features_with_label)))


def list_audio_files(dir: str) -> np.ndarray:
    files = list(map(lambda path: path.replace("\\", PATH_SEPARATOR),
                     filter(lambda path: path.endswith(AUDIO_EXT),
                            [os.path.join(dp, f) for dp, dn, fn in os.walk(dir) for f in fn])))

    return np.asarray(files)


def create_gender_file() -> None:
    def id_gender_tuple(line: str) -> (str, int):
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
    global gender_dict
    if gender_dict is None:
        def id_gender_tuple(line):
            t = line.split(GENDERS_FILE_SEPARATOR)
            id = t[0]
            gender = int(t[1])
            return id, gender

        fd = open(GENDERS_FILE, "r")
        gender_dict = dict(map(lambda line: id_gender_tuple(line), [line.rstrip() for line in fd]))
    return gender_dict
