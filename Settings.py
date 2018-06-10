import multiprocessing

SEED = 1  # Seed for random
N_JOBS = multiprocessing.cpu_count() - 1  # Number of jobs for multithreading methods
NB_CLASSES = 2  # Two genders
TRAIN_PERCENT = 0.8  # Percentage of files to use for training
CV = 10  # Number of cross-validation splits to create
FEATURES_NUMBER = 20  # Number of features to extract from MFCC
FEATURES_WINDOW_SIZE = 10  # Found by checking the shape

# NN settings
VALIDATION_PERCENT = 0.2  # Percentage of files to use for validation
BATCH_SIZE = 128  # batch size for NN
NUM_EPOCHS = 300  # Max number of epochs for NN
FILTER_DEPTH = 32  # Filter depth for Conv2D
KERNEL_SIZE = (3, 3)  # Kernel size for Conv2D
LEARNING_RATE = 0.001  # Learning rate for NN
STRIDES = (1, 1)  # Strides for Convolution
INPUT_SHAPE = (FEATURES_WINDOW_SIZE, FEATURES_NUMBER, 1)  # Input shape for CNN
DATA_FORMAT = "channels_last"  # Input data format

# Various directories for the project
MODELS_DIR = "models/"
OUTPUT_DIR = "output/"
DATA_DIR = "data/"
TENSORBOARD_DIR = "tensorboard/"

# File settings
LIBRISPEECH_DIR = DATA_DIR + "LibriSpeech/"
SPEAKERS_FILE = LIBRISPEECH_DIR + "SPEAKERS.TXT"  # The speakers file containing the gender infos
AUDIO_FILES_DIR = LIBRISPEECH_DIR + "dev-clean/"
AUDIO_EXT = "flac"
GENDERS_FILE = DATA_DIR + "Genders.txt"  # The generated genders file for easier retrieving
FEATURES_WITH_LABEL_FILE = DATA_DIR + "Features_with_label.npy"  # File the features along with their labels will be saved to
MIN_FEATURES_FILE = DATA_DIR + "Min.npy"  # In case one would want to predict a new file
MAX_FEATURES_FILE = DATA_DIR + "Max.npy"  # In case one would want to predict a new file
FILE_ID_SEPARATOR = "-"
PATH_SEPARATOR = "/"
DUMP_EXT = ".pkl"

# Used to parse and generate genders file
F_LABEL = 0
M_LABEL = 1
COMMENT_STARTER = ";"
SPEAKERS_F_LABEL = "F"
SPEAKERS_M_LABEL = "M"
SPEAKERS_FILE_SEPARATOR = "|"
GENDERS_FILE_SEPARATOR = ","
