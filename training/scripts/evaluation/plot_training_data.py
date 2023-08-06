import numpy as np
from training.modules.utils.plotting.plotting import plot_mse_history

# load train history files from checkpoint
eval_mse_history_1 = np.load("/home/rebecca/Downloads/datatoplot/RATNeck_LSTM6_BS1_ep181/training_data/eval_mse.npy")
eval_mse_history_2 = np.load("/home/rebecca/Downloads/datatoplot/BluesDriverLSTM6BL1_ep199/training_data/eval_mse.npy")
eval_mse_history_3 = np.load("/home/rebecca/Downloads/datatoplot/TubeScreamer_LSTM6_BS1_ep192/training_data/eval_mse.npy")
#train_mse_history = np.load("/home/rebecca/Downloads/datatoplot/RATNeck_LSTM6_BS1_ep181/training_data/train_mse.npy")
training_time = np.load("/home/rebecca/Downloads/datatoplot/RATNeck_LSTM6_BS1_ep181/training_data/training_time.npy")
#print(np.shape(eval_mse_history))
#print(np.shape(train_mse_history))
#print(np.shape(training_time))
#print(eval_mse_history)
#print(train_mse_history)
#print(training_time)

# normalize training data if needed


# plot training data
plot_mse_history(eval_mse_history_1, eval_mse_history_2, eval_mse_history_3)


