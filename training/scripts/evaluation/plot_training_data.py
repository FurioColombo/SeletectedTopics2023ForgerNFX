import numpy as np
from training.modules.utils.plotting.plotting import plot_mse_history

# load train history files from checkpoint
eval_mse_history_1 = np.load("/home/rebecca/Downloads/datatoplot/RATNeck_LSTM6_BS1_ep181/training_data/eval_mse.npy")
eval_mse_history_2 = np.load("/home/rebecca/Downloads/datatoplot/BluesDriverLSTM6BL1_ep199/training_data/eval_mse.npy")
eval_mse_history_3 = np.load("/home/rebecca/Downloads/datatoplot/TubeScreamer_LSTM6_BS1_ep192/training_data/eval_mse.npy")

eval_mse_history_1_16 = np.load("/home/rebecca/Downloads/datatoplot/RATNeck_single_notes_bl16lstm8/_ep887/training_data/eval_mse.npy")
eval_mse_history_2_16 = np.load("/path/to/file")
eval_mse_history_3_16 = np.load("/path/to/file") 

# plot training data
plot_mse_history(eval_mse_history_1, eval_mse_history_2, eval_mse_history_3, eval_mse_history_1_16, eval_mse_history_2_16, eval_mse_history_3_16)


