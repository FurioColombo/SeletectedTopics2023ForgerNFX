import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from _datetime import datetime
from training.config import config

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from training.modules.data.generators.egfx_generator import EGFxDatasetGenerator
from training.modules.data.generators.fragments_generator import FragmentsDatasetGenerator
from training.modules.utils.plotting.plotting import plot_dataset_couple
import os


# used for the writing of example outputs
run_name = config.DATASET_TARGET_FOLDER_NAME.split('/')[1]
date_time = datetime.now().strftime("_%m-%d-%Y_%H-%M")

assert os.path.exists(config.DATASET_FOLDER_PATH), "Audio folder  not found. Looked for " + config.DATASET_FOLDER_PATH

# used to render example output during training
assert os.path.exists(config.TEST_FILE_PATH), "Test file not found. Looked for " + config.TEST_FILE_PATH

# create the logger for tensorboard
writer = SummaryWriter()

print("Loading dataset from folder ", config.DATASET_FOLDER_PATH)

# if any(config.DATASET_TYPE.lower() == name for name in ['egfx', 'single_notes']):
dataset_generator_EGFX = EGFxDatasetGenerator(
    input_audio_folder=config.DATASET_FOLDER_PATH + config.DATASET_INPUT_FOLDER_NAME,
    output_audio_folder=config.DATASET_FOLDER_PATH + config.DATASET_TARGET_FOLDER_NAME,
    samplerate=config.SAMPLE_RATE,
    normalize_amp=True,
)
# elif any(config.DATASET_TYPE.lower() == name for name in ['myk', 'fragments']):
dataset_generator = FragmentsDatasetGenerator(
    input_audio_folder=config.DATASET_FOLDER_PATH + config.DATASET_INPUT_FOLDER_NAME,
    output_audio_folder=config.DATASET_FOLDER_PATH + config.DATASET_TARGET_FOLDER_NAME,
    samplerate=config.SAMPLE_RATE,
    frag_len_seconds=2
)
# else:
#    assert 'ERROR: config dataset name not matching any available option'

print("Generating dataset...", end='')
dataset = dataset_generator.generate_dataset()
dataset_EGFX = dataset_generator_EGFX.generate_dataset()

# note that if the dataset is shuffled each time it's generated, we are not getting deterministic samples
# plot_dataset_couple(dataset_EGFX, random_sample=True)
plot_dataset_couple(dataset, random_sample=True)

'''
dataset = (np.load('C:/Users/Marco Furio Colombo/Desktop/myk_dataset_0.npy'), np.load('C:/Users/Marco Furio Colombo/Desktop/myk_dataset_1.npy'))

dry_samples = dataset[0]
wet_samples = dataset[1]

plt.figure()
sns.set_theme()
sample_len = dry_samples.shape[1]
dataset_index = np.random.randint(0, dry_samples.shape[0])

dry_sample = dry_samples[dataset_index, :, :]
wet_sample = wet_samples[dataset_index, :, :]

print('type(dry_sample):', type(dry_sample))
print('dry_sample.shape:', dry_sample.shape)
print('dry_sample len:', sample_len)
x = np.linspace(0, int(sample_len/ config.SAMPLE_RATE), sample_len)

dry_sample = dry_sample.flatten()
wet_sample = wet_sample.flatten()

if (np.square(wet_sample)).mean() > (np.square(dry_sample)).mean():
    sns.lineplot(x=x, y=wet_sample, color='blue')
    sns.lineplot(x=x, y=dry_sample, color='red')
else:
    sns.lineplot(x=x, y=dry_sample, color='red')
    sns.lineplot(x=x, y=wet_sample, color='blue')

plt.show()
'''