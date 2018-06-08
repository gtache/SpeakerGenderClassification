SEED = 1  # Seed for random
NB_CLASSES = 2  # Two genders
TRAIN_PERCENT = 0.8  # Percentage of files to use for training
VALIDATION_PERCENT = 0.2  # Percentage of files to use for validation
BATCH_SIZE = 128  # batch size for NN
NUM_EPOCHS = 100  # Number of epochs for NN
FILTER_DEPTH = 32  # Filter depth for CNN
KERNEL_SIZE = (3, 3)  # Kernel size for Convolution
LEARNING_RATE = 0.001  # Learning rate for NN
STRIDES = (1, 1)  # Strides for Convolution
DATA_FORMAT = "channels_last"  # Input data format

# Different directories for the project
MODELS_DIR = "models/"
OUTPUT_DIR = "output/"
DATA_DIR = "data/"
TENSORBOARD_DIR = "tensorboard/"

LIBRISPEECH_DIR = DATA_DIR + "LibriSpeech/"
SPEAKERS_FILE = LIBRISPEECH_DIR + "SPEAKERS.TXT"  # The speakers file containing the gender infos
AUDIO_FILES_DIR = LIBRISPEECH_DIR + "dev-clean/"
AUDIO_EXT = "flac"
GENDERS_FILE = DATA_DIR + "Genders.txt"  # The generated genders file for easier retrieving
TWO_D_FEATURES_WITH_LABELS_FILE = DATA_DIR + "2DFeaturesWithLabel.npy"  # The file the 2D data will be saved to
TEMPORAL_TWO_D_FEATURES_WITH_LABELS_FILE = DATA_DIR + "Temporal2DFeaturesWithLabel.npy"  # The file the Temporal 2D data will be saved to
ONE_D_FEATURES_WITH_LABELS_FILE = DATA_DIR + "1DFeaturesWithLabel.npy"  # The file the 1D data will be saved to
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

FEATURES_NUMBER = 20  # Number of features to extract from MFCC
MIN_SAMPLES_PER_FILE = 46  # Found by checking the shape
INPUT_SHAPE = (MIN_SAMPLES_PER_FILE, FEATURES_NUMBER, 1)  # Input shape for CNN
