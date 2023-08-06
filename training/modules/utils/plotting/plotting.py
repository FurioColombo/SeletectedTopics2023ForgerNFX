import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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


def plot_mse_history(eval_mse_history_1, eval_mse_history_2, eval_mse_history_3):

    # setup data
    """epochs_train = train_mse_history[:,0]
    mse_train_values = train_mse_history[:, 1]"""
    epochs_eval_1 = eval_mse_history_1[:,0]
    mse_eval_values_1 = eval_mse_history_1[:,1] #RAT
    epochs_eval_2 = eval_mse_history_2[:,0]
    mse_eval_values_2 = eval_mse_history_2[:,1] #BluesDriver
    epochs_eval_3 = eval_mse_history_3[:,0]
    mse_eval_values_3 = eval_mse_history_3[:,1] #TubeScreamer

    # setup dataframes for Seaborn
    """data_train = {
        'Epochs': epochs_train,
        'MSE Train': mse_train_values
    }
    df_train = pd.DataFrame(data_train)"""

    #RAT dataframe
    data_eval_1 = {
    'Epochs': epochs_eval_1,
    'MSE Evaluation': mse_eval_values_1
    }
    df_eval_1 = pd.DataFrame(data_eval_1)

    #BluesDriver dataframe
    data_eval_2 = {
    'Epochs': epochs_eval_2,
    'MSE Evaluation': mse_eval_values_2
    }
    df_eval_2 = pd.DataFrame(data_eval_2)

    #TubeScreamer dataframe
    data_eval_3 = {
    'Epochs': epochs_eval_3,
    'MSE Evaluation': mse_eval_values_3
    }
    df_eval_3 = pd.DataFrame(data_eval_3)

    plt.figure(figsize=(20,12))
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.lineplot(data=df_eval_1, x='Epochs', y='MSE Evaluation', label='MSE Evaluation RAT', marker='o')
    sns.lineplot(data=df_eval_2, x='Epochs', y='MSE Evaluation', label='MSE Evaluation BluesDriver', marker='o')
    sns.lineplot(data=df_eval_3, x='Epochs', y='MSE Evaluation', label='MSE Evaluation TubeScreamer', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE History')
    plt.legend()
    plt.grid(True)
    plt.show()

