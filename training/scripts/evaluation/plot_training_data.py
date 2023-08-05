import numpy as np
from training.modules.utils.plotting.plotting import plot_mse_history

# load train history files from checkpoint
eval_mse_history = np.load(".........\resources\checkpoints\BluesDriverNeck_egfx_bl1lstm7\_ep11\training_data\eval_mse.npy")
train_mse_history = np.load(".........\resources\checkpoints\BluesDriverNeck_egfx_bl1lstm7\_ep11\training_data\train_mse.npy")
print(np.shape(eval_mse_history))
print(np.shape(eval_mse_history))

# normalize training data if needed


# plot training data
plot_mse_history(eval_mse_history)
plot_mse_history(train_mse_history)

