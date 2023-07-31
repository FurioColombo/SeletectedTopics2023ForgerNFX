import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from training.config import config


def plot_dataset_couple(dataset, dataset_index=0, random_sample=False):
    dry_samples = dataset.tensors[0]
    wet_samples = dataset.tensors[1]
    num_samples = dry_samples.shape[0]
    sample_len = dry_samples.shape[1]
    print(dry_samples.shape)
    print(wet_samples.shape)

    if random_sample:
        dataset_index = np.random.randint(0, num_samples)

    plt.figure()
    sns.set_theme()

    dry_sample = dry_samples[dataset_index, :, :]
    wet_sample = wet_samples[dataset_index, :, :]

    x = np.linspace(0, int(sample_len / config.SAMPLE_RATE), sample_len)

    dry_sample = torch.flatten(dry_sample)
    wet_sample = torch.flatten(wet_sample)
    print(dry_sample.shape)
    print(wet_sample.shape)
    print(x.shape)
    if (np.square(wet_sample)).mean() > (np.square(dry_sample)).mean():
        sns.lineplot(x=x, y=wet_sample, color='blue')
        sns.lineplot(x=x, y=dry_sample, color='red')
    else:
        sns.lineplot(x=x, y=dry_sample, color='red')
        sns.lineplot(x=x, y=wet_sample, color='blue')

    plt.show()

