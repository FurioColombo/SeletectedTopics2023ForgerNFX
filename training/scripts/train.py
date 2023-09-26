## This script trains an LSTM according
## to the method described in
## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from _datetime import datetime
import torch
import os

from training.modules.data.generators.egfx_generator import EGFxDatasetGenerator
from training.modules.data.generators.fragments_generator import FragmentsDatasetGenerator
from training.modules.training import training
from training.modules.models import lstm_model, conv_model
from training.modules.training.trainer import Trainer
from training.config import config


# used for the writing of example outputs
run_name = config.DATASET_TARGET_FOLDER_NAME[0].split('/')[1]
date_time = datetime.now().strftime("_%m-%d-%Y_%H-%M")

assert os.path.exists(config.DATASET_FOLDER_PATH), "Audio folder  not found. Looked for " + config.DATASET_FOLDER_PATH
# used to render example output during training
assert os.path.exists(config.TEST_FILE_PATH), "Test file not found. Looked for " + config.TEST_FILE_PATH

# create the logger for tensorboard
writer = SummaryWriter()

print("Loading dataset from folder ", config.DATASET_FOLDER_PATH)

if any(config.DATASET_TYPE.lower() == name for name in ['egfx', 'single_notes']):
    in_audio_folders = [config.DATASET_FOLDER_PATH + folder_name for folder_name in config.DATASET_INPUT_FOLDER_NAME]
    out_audio_folders = [config.DATASET_FOLDER_PATH + folder_name for folder_name in config.DATASET_TARGET_FOLDER_NAME]
    dataset_generator = EGFxDatasetGenerator(
        input_audio_folders=in_audio_folders,
        output_audio_folders=out_audio_folders,
        block_size=config.NN_IN_BLOCK_SIZE,
        samplerate=config.SAMPLE_RATE,
        normalize_amp=True,
    )
elif any(config.DATASET_TYPE.lower() == name for name in ['myk', 'fragments']):
    dataset_generator = FragmentsDatasetGenerator(
        input_audio_folder=config.DATASET_FOLDER_PATH + config.DATASET_INPUT_FOLDER_NAME,
        output_audio_folder=config.DATASET_FOLDER_PATH + config.DATASET_TARGET_FOLDER_NAME,
        block_size=config.NN_IN_BLOCK_SIZE,
        samplerate=config.SAMPLE_RATE,
        frag_len_seconds=1
    )
else:
    assert 'ERROR: config dataset name not matching any available option'

print("Generating dataset...", end='')
dataset = dataset_generator.generate_dataset()
print("Done\nSplitting dataset...", end='')
train_ds, val_ds, test_ds = dataset_generator.get_train_valid_test_datasets()

print("Done\n Looking for GPU power")
device = training.get_device()

print("Creating data loaders")
train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))
val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))
test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=True, generator=torch.Generator(device=device))

print("Creating model")
if config.MODEL_NAME.lower() == 'lstm'.lower():
    model = lstm_model.SimpleLSTM(
        hidden_size=config.LSTM_HIDDEN_SIZE,
        samples_to_process=config.NN_IN_BLOCK_SIZE
    ).to(device)
elif config.MODEL_NAME.lower() == 'conv'.lower():
    model = conv_model.SimpleConv1d(
        kernel_size=config.KERNEL_SIZE,
        normalize_output=True
    ).to(device)
else:
    model = None
assert model is not None
if config.MODEL_CHECKPOINT is not None:
    print('loading model checkpoint... ', end='')
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT)['model_state_dict'])
    print('Done')

trainer = Trainer(
    model=model,
    writer=writer,
    run_name=run_name
)

trainer.train_model(
    dataloader=train_dl,
)
