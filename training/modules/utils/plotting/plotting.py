import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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


def plot_mse_history(eval_mse_history_1, eval_mse_history_2, eval_mse_history_3, eval_mse_history_1_16, eval_mse_history_2_16, eval_mse_history_3_16):
    # setup data
    epochs_eval_1 = eval_mse_history_1[:,0] #RAT
    mse_eval_values_1 = eval_mse_history_1[:,1] 
    epochs_eval_2 = eval_mse_history_2[:,0] #BluesDriver
    mse_eval_values_2 = eval_mse_history_2[:,1] 
    epochs_eval_3 = eval_mse_history_3[:,0] #TubeScreamer
    mse_eval_values_3 = eval_mse_history_3[:,1] 

    epochs_eval_1_16 = eval_mse_history_1_16[:,0] #RAT bs16
    mse_eval_values_1_16 = eval_mse_history_1_16[:,1]
    epochs_eval_2_16 = eval_mse_history_2_16[:,0] #BluesDriver bs16
    mse_eval_values_2_16 = eval_mse_history_2_16[:,1]
    epochs_eval_3_16 = eval_mse_history_3_16[:,0] #TubeScreamer bs16
    mse_eval_values_3_16 = eval_mse_history_3_16[:,1]
    
    # setup dataframes for Seaborn

    #RAT dataframe
    data_eval_1 = {
    'Epochs': epochs_eval_1,
    'MSE Evaluation': mse_eval_values_1,
    }
    df_eval_1 = pd.DataFrame(data_eval_1)

    #RAT dataframe bs16
    data_eval_1_16 = {
        'Epochs':epochs_eval_1_16,
        'MSE Evaluation': mse_eval_values_1_16
    }
    df_eval_1_16 = pd.DataFrame(data_eval_1_16)

    #BluesDriver dataframe
    data_eval_2 = {
    'Epochs': epochs_eval_2,
    'MSE Evaluation': mse_eval_values_2
    }
    df_eval_2 = pd.DataFrame(data_eval_2)

    #BluesDriver dataframe bs16
    data_eval_2_16 = {
    'Epochs': epochs_eval_2_16,
    'MSE Evaluation': mse_eval_values_2_16
    }
    df_eval_2_16 = pd.DataFrame(data_eval_2_16)

    #TubeScreamer dataframe
    data_eval_3 = {
    'Epochs': epochs_eval_3,
    'MSE Evaluation': mse_eval_values_3
    }
    df_eval_3 = pd.DataFrame(data_eval_3)

    #TubeScreamer dataframe bs16
    data_eval_3_16 = {
    'Epochs': epochs_eval_3_16,
    'MSE Evaluation': mse_eval_values_3_16
    }
    df_eval_3_16 = pd.DataFrame(data_eval_3_16)

    plt.figure(figsize=(20,12))
    sns.set_theme(style="whitegrid", palette="coolwarm")
    sns.lineplot(data=df_eval_1, x='Epochs', y='MSE Evaluation', label='MSE Evaluation RAT', marker='o')
    sns.lineplot(data=df_eval_1_16, x='Epochs', y='MSE Evaluation', label='MSE Evaluation RAT bs 16', marker='o')
    sns.lineplot(data=df_eval_2, x='Epochs', y='MSE Evaluation', label='MSE Evaluation BluesDriver', marker='o')
    sns.lineplot(data=df_eval_2_16, x='Epochs', y='MSE Evaluation', label='MSE Evaluation BluesDriver bs 16', marker='o')
    sns.lineplot(data=df_eval_3, x='Epochs', y='MSE Evaluation', label='MSE Evaluation TubeScreamer', marker='o')
    sns.lineplot(data=df_eval_3_16, x='Epochs', y='MSE Evaluation', label='MSE Evaluation TubeScreamer bs 16', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE History')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mse_datasets(eval_mse_history_1, eval_mse_history_2):
    # setup data
    epochs_eval_1 = eval_mse_history_1[:,0] #RAT EGFX dataset
    mse_eval_values_1 = eval_mse_history_1[:,1] 
    epochs_eval_2 = eval_mse_history_2[:,0] #RAT Fragments dataset
    mse_eval_values_2 = eval_mse_history_2[:,1] 

    # normalization with MinMax
    scaler_epochs = MinMaxScaler(feature_range=(0,200))
    scaler_mse = MinMaxScaler(feature_range=(0,0.8))

    #RAT
    epochs_eval_1_norm=scaler_epochs.fit_transform(epochs_eval_1.reshape(-1,1)).flatten()
    mse_eval_values_1_norm=scaler_mse.fit_transform(mse_eval_values_1.reshape(-1,1)).flatten()

    epochs_eval_2_norm=scaler_epochs.fit_transform(epochs_eval_2.reshape(-1,1)).flatten()
    mse_eval_values_2_norm=scaler_mse.fit_transform(mse_eval_values_2.reshape(-1,1)).flatten()

    # setup dataframes for Seaborn

    #RAT dataframe EGFX
    data_eval_1 = {
        'Epochs': epochs_eval_1_norm,
        'MSE Evaluation': mse_eval_values_1_norm,
    }
    df_eval_1 = pd.DataFrame(data_eval_1)

    #RAT dataframe Fragments
    data_eval_2 = {
        'Epochs':epochs_eval_2_norm,
        'MSE Evaluation': mse_eval_values_2_norm
    }
    df_eval_2 = pd.DataFrame(data_eval_2)

    plt.figure(figsize=(20,12))
    sns.set_theme(style="whitegrid", palette="coolwarm")
    sns.lineplot(data=df_eval_1, x='Epochs', y='MSE Evaluation', label='MSE Evaluation RAT EGFX', marker='o')
    sns.lineplot(data=df_eval_2, x='Epochs', y='MSE Evaluation', label='MSE Evaluation RAT Fragments', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE History')
    plt.legend()
    plt.grid(True)
    plt.show()


