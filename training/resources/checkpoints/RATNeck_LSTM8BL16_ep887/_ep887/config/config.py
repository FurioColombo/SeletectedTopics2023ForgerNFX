# TODO: move config options to .ini file to avoid the need for changes without the need for any rebuild

# ====================== PATHS ======================
DATASET_INPUT_FOLDER_NAME = '/Clean/Neck'
DATASET_TARGET_FOLDER_NAME = '/RAT/Neck'
# DATASET_TARGET_FOLDER_NAME = '/NeumannSaxoTenor'
OUTPUT_FOLDER_PATH = '/home/rebecca/Desktop/SeletectedTopics2023ForgerNFX/training/resources'
DATASET_FOLDER_PATH = "/home/rebecca/Desktop/SeletectedTopics2023ForgerNFX/training/resources/EGFxDataset"
MODEL_CHECKPOINT = None
TEST_FILE_PATH = "/home/rebecca/Desktop/SELECTED_TOPICS/Polimi_starter_v4/data/guitar.wav"

# ====================== DATASET ======================
SAMPLE_RATE = 44100
DATASET_TYPE = 'single_notes' # `single_notes`, `egfx`, `fragments`,  `myk`

# ====================== MODEL ======================
MODEL_NAME = 'lstm'
LSTM_HIDDEN_SIZE = 8
KERNEL_SIZE = 8
NN_IN_BLOCK_SIZE = 16

# ====================== TRAINING ======================
LEARNING_RATE = 5e-3
BATCH_SIZE = 3
MAX_EPOCHS = 901  # +1 needed for checkpoint savings
SAVE_EVERY_N_BATCHES = 10
EVAL_EACH_N_EPOCHS = 10
















