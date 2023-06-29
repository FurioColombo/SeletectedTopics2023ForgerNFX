import evaluate
from modules.models import models
from modules.training import training
from config import config
import torch

device = training.get_device()
model = models.SimpleLSTM(hidden_size=config.LSTM_HIDDEN_SIZE).to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)
checkpoint = torch.load("C:/Users/Marco Furio Colombo/Desktop/Polimi - MAE/Selected Topics/Polimi_starter_v4/035c_train_lstm/furio_python/EGFx/scripts/checkpoints/pytorch/BluesDriver/Neck/_05-22-2023_18-10/model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

evaluate.run_file_through_model(model,
                                config.TEST_FILE_PATH,
                                config.AUDIO_FOLDER_PATH +
                                "data/EGFxDataset/Clean/Middle/2-10.wav")
