# TODO: move config options to .ini file to avoid the need for changes without the need for any rebuild
import os
from training.modules.utils.file_system import find_vcs_root

# ====================== PATHS ======================
PROJECT_ROOT = find_vcs_root(__file__)
TRAINING_ROOT = os.path.join(PROJECT_ROOT, 'training')

DATASET_INPUT_FOLDER_NAME = ['/Clean/Neck', '/Clean/Middle-Neck', '/Clean/Middle', '/Clean/Bridge-Middle', '/Clean/Bridge']
DATASET_TARGET_FOLDER_NAME = ['/RAT/Neck', '/RAT/Middle-Neck', '/RAT/Middle', '/RAT/Bridge-Middle', '/RAT/Bridge']
DATASET_FOLDER_PATH = os.path.join(TRAINING_ROOT, 'resources', 'datasets', 'EGFxDataset')

# DATASET_FOLDER_PATH = 'C:/Users/Marco Furio Colombo/Desktop/Polimi - MAE/Selected Topics/materials/Polimi_starter_v4/data/audio_ht1'
# DATASET_INPUT_FOLDER_NAME = '/input/wav'
# DATASET_TARGET_FOLDER_NAME = '/output/wav'
# DATASET_TARGET_FOLDER_NAME = '/NeumannSaxoTenor'


OUTPUT_FOLDER_PATH = os.path.join(TRAINING_ROOT, 'resources', 'testing')
TEST_FILE_PATH = os.path.join(TRAINING_ROOT, 'resources', 'datasets', 'guitar.wav')
# MODEL_CHECKPOINT = "C:/Users/Marco Furio Colombo/Desktop/forger_nfx/training/resources/checkpoints/RATNeck_egfx_bl16lstm64/_ep112/model_weights/pytorch/model.pt"
MODEL_CHECKPOINT = None
START_EPOCH = 0

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
MAX_EPOCHS = 2001  # +1 for checkpoint savings
SAVE_EVERY_N_BATCHES = 10
EVAL_EACH_N_EPOCHS = 10

# ====================== EVALUATION ======================
















