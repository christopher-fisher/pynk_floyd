# Constants for use in the pynk_floyd project

# File path for training data
# TRAINING_DATA_PATH = "C:\\Users\\daeur\\PycharmProjects\\pynk_floyd\\Training Data\\darkside.txt"
TRAINING_DATA_PATH = "C:\\Users\\daeur\\PycharmProjects\\pynk_floyd\\Training Data\\pinkfloyd.txt"

EPOCHS = 25

# Lower = more predictable, higher = more surprising.
# This is a good number to tweak
TEMPERATURE = 1.0

# List of initial inputs for batch generation
# This will change after the proof of concept stage
# OUTPUT_SEEDS = ['D', 'A', 'R', 'K', 'S', 'I', 'D', 'E']
# OUTPUT_SEEDS = ['D', 'A', 'D']
OUTPUT_SEEDS = ['P', 'I', 'N', 'K', 'F', 'L', 'O', 'Y', 'D']

# Name to use in folder creation
PROJECT_NAME = 'pinkfloyd'