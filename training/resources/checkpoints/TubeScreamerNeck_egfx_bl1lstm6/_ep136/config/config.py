# TODO: move config options to .ini file to avoid the need for changes without the need for any rebuild

# ====================== PATHS ======================

DATASET_INPUT_FOLDER_NAME = '/Clean/Neck'
DATASET_TARGET_FOLDER_NAME = '/TubeScreamer/Neck'
DATASET_FOLDER_PATH = "C:/Users/Marco Furio Colombo/Desktop/Polimi - MAE/Selected Topics/materials/Polimi_starter_v4/data/EGFxDataset"
# DATASET_FOLDER_PATH = 'C:/Users/Marco Furio Colombo/Desktop/Polimi - MAE/Selected Topics/materials/Polimi_starter_v4/data/audio_ht1'
# DATASET_INPUT_FOLDER_NAME = '/input/wav'
# DATASET_TARGET_FOLDER_NAME = '/output/wav'
# DATASET_TARGET_FOLDER_NAME = '/NeumannSaxoTenor'
OUTPUT_FOLDER_PATH = 'C:/Users/Marco Furio Colombo/Desktop/forger_nfx/training/resources'
TEST_FILE_PATH = "C:/Users/Marco Furio Colombo/Desktop/Polimi - MAE/Selected Topics/materials/Polimi_starter_v4/data/guitar.wav"
# MODEL_CHECKPOINT = "C:/Users/Marco Furio Colombo/Desktop/forger_nfx/training/resources/checkpoints/TubeScreamerNeck_egfx_bl1lstm7/_ep69/model_weights/pytorch/model.pt"
MODEL_CHECKPOINT = None

# ====================== DATASET ======================
SAMPLE_RATE = 44100
DATASET_TYPE = 'egfx'

# ====================== MODEL ======================
MODEL_NAME = 'lstm'
LSTM_HIDDEN_SIZE = 6
KERNEL_SIZE = 8
NN_IN_BLOCK_SIZE = 1

# ====================== TRAINING ======================
LEARNING_RATE = 5e-3
BATCH_SIZE = 4
MAX_EPOCHS = 201  # +1 for checkpoint savings
SAVE_EVERY_N_BATCHES = 10
EVAL_EACH_N_EPOCHS = 10

















