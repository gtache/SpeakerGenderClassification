SEED = 1  # Seed for random
NB_CLASSES = 2  # Two genders
NUM_FILES = 1  # Number of audio files
TRAIN_PERCENT = 0.8  # Percentage of files to use for training
VALIDATION_PERCENT = 0.2  # Percentage of files to use for validation
BATCH_SIZE = 128  # batch size for NN
NUM_EPOCHS = 100
FILTER_DEPTH = 32
LEARNING_RATE = 0.001

MODELS_DIR = "models/"
OUTPUT_DIR = "output/"
DATA_DIR = "data/"

LIBRISPEECH_DIR = DATA_DIR + "LibriSpeech/"
TENSORBOARD_DIR = "tensorboard/"
SPEAKERS_FILE = LIBRISPEECH_DIR + "SPEAKERS.TXT"
AUDIO_FILES_DIR = LIBRISPEECH_DIR + "dev-clean/"
AUDIO_EXT = "flac"
GENDERS_FILE = DATA_DIR + "Genders.txt"
TWO_D_FEATURES_WITH_LABELS_FILE = DATA_DIR + "TwoDFeaturesWithLabel.npy"
ONE_D_FEATURES_WITH_LABELS_FILE = DATA_DIR + "OneDFeaturesWithLabel.npy"
FILE_ID_SEPARATOR = "-"
PATH_SEPARATOR = "/"
DUMP_EXT = ".pkl"

F_LABEL = 0
M_LABEL = 1
COMMENT_STARTER = ";"
SPEAKERS_F_LABEL = "F"
SPEAKERS_M_LABEL = "M"
SPEAKERS_FILE_SEPARATOR = "|"
GENDERS_FILE_SEPARATOR = ","

FEATURES_NUMBER = 20
MAX_SAMPLES_FOR_FILES = 1021  # Found by checking the shape
INPUT_SHAPE = (FEATURES_NUMBER, MAX_SAMPLES_FOR_FILES)
