import numpy as np
import os
from training.modules.utils.plotting.plotting import plot_mse_history, plot_mse_datasets

# load train history files from checkpoint
eval_mse_history_1 = np.load('datatoplot/RATNeck_LSTM6BS1_ep181.npy') #RAT BS1
eval_mse_history_2 = np.load('datatoplot/BluesDriver_LSTM6BS1_ep199.npy') #BluesDriver BS1
eval_mse_history_3 = np.load('datatoplot/TubeScreamer_LSTM6BS1_ep192.npy') #TubeScreamer BS1

eval_mse_history_1_16 = np.load('datatoplot/RATNeck_LSTM64BS16_ep196.npy') #RAT BS16
eval_mse_history_2_16 = np.load('datatoplot/BluesDriver_LSTM64BS16_ep397.npy') #BluesDriver BS16
eval_mse_history_3_16 = np.load('datatoplot/TubeScreamer_LSTM64BS16_ep689.npy') #TubeScreamer BS16

# plot training data for MSE evaluation (same dataset)
plot_mse_history(eval_mse_history_1, eval_mse_history_2, eval_mse_history_3, eval_mse_history_1_16, eval_mse_history_2_16, eval_mse_history_3_16)

# plot training data for MSE evaluation (compare different datasets)
eval_mse_history_fr = np.load("datatoplot/RATNeck_LSTM6BS1_Fragments.npy") #load eval file.npy for the Fragment dataset
plot_mse_datasets(eval_mse_history_1, eval_mse_history_fr)