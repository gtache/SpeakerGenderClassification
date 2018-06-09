SEED = 1  # Seed for random
NB_CLASSES = 2  # Two genders
TRAIN_PERCENT = 0.8  # Percentage of files to use for training
FEATURES_NUMBER = 20  # Number of features to extract from MFCC
MIN_SAMPLES_PER_FILE = 46  # Found by checking the shape

# NN settings
VALIDATION_PERCENT = 0.2  # Percentage of files to use for validation
BATCH_SIZE = 128  # batch size for NN
NUM_EPOCHS = 100  # Number of epochs for NN
FILTER_DEPTH = 32  # Filter depth for CNN
KERNEL_SIZE = (3, 3)  # Kernel size for Convolution
LEARNING_RATE = 0.001  # Learning rate for NN
STRIDES = (1, 1)  # Strides for Convolution
INPUT_SHAPE = (MIN_SAMPLES_PER_FILE, FEATURES_NUMBER, 1)  # Input shape for CNN
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
