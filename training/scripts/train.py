## This script trains an LSTM according
## to the method described in 
## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from _datetime import datetime
from config import config
from modules.models import conv_model
import evaluate

from modules.data.generators.egfx_generator import EGFxDatasetGenerator
from modules.data.generators.fragments_generator import FragmentsDatasetGenerator
from modules.training import training
from modules.models import lstm_model
from modules.models import loss
import torch
import os


# used for the writing of example outputs
run_name = config.DATASET_TARGET_FOLDER_NAME.split('/')[1]
date_time = datetime.now().strftime("_%m-%d-%Y_%H-%M")

assert os.path.exists(config.AUDIO_FOLDER_PATH), "Audio folder  not found. Looked for " + config.AUDIO_FOLDER_PATH
# used to render example output during training
assert os.path.exists(config.TEST_FILE_PATH), "Test file not found. Looked for " + config.TEST_FILE_PATH


# create the logger for tensorboard
writer = SummaryWriter()

print("Loading dataset from folder ", config.AUDIO_FOLDER_PATH)

if any(config.DATASET_TYPE.lower() == name for name in ['egfx', 'single_notes']):
    dataset_generator = EGFxDatasetGenerator(
        input_audio_folder=config.AUDIO_FOLDER_PATH + config.DATASET_INPUT_FOLDER_NAME,
        output_audio_folder=config.AUDIO_FOLDER_PATH + config.DATASET_TARGET_FOLDER_NAME,
        samplerate=config.SAMPLE_RATE,
        normalize_amp=True,
    )
elif any(config.DATASET_TYPE.lower() == name for name in ['myk', 'fragments']):
    dataset_generator = FragmentsDatasetGenerator(
        input_audio_folder=config.AUDIO_FOLDER_PATH + config.DATASET_INPUT_FOLDER_NAME,
        output_audio_folder=config.AUDIO_FOLDER_PATH + config.DATASET_TARGET_FOLDER_NAME,
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
    model = lstm_model.SimpleLSTM(hidden_size=config.LSTM_HIDDEN_SIZE).to(device)
elif config.MODEL_NAME.lower() == 'conv'.lower():
    model = conv_model.SimpleConv1d(
        kernel_size=config.KERNEL_SIZE,
        normalize_output=True
    )

print("Creating optimiser")
# https://github.com/Alec-Wright/Automated-GuitarAmpModelling/blob/main/dist_model_recnet.py
optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)
print("Creating loss functions")
# https://github.com/Alec-Wright/CoreAudioML/blob/bad9469f94a2fa63a50d70ff75f5eff2208ba03f/training.py
loss_functions = loss.LossWrapper()

# now the training loop
print("About to train")
lowest_val_loss = 0
best_loss = False
for epoch in range(config.MAX_EPOCHS):
    ep_loss = training.train_epoch_interval(
        model,
        train_dl,
        loss_functions,
        optimiser,
        device=device,
        sub_batch_seq_len=-1,
        warm_up_len=1000
    )

    # ep_loss = training.train_epoch(model, train_dl, loss_functions, optimiser, device=device)
    val_loss = training.compute_batch_loss(model, val_dl, loss_functions, device=device)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Loss/train", ep_loss, epoch)

    # check if we have beaten our best loss to date
    if lowest_val_loss == 0:  # first run
        lowest_val_loss = val_loss
    elif val_loss < lowest_val_loss:  # new record
        lowest_val_loss = val_loss
        best_loss = True
    else:  # no improvement
        best_loss = False
    if best_loss:  # save best model so far
        print("Record loss - saving at ", epoch)
        # save fro RTNeural
        checkpoint_path = model.save_model(
            out_dir=os.path.join(config.OUTPUT_FOLDER_PATH),
            curr_epoch=epoch,
            optimizer=optimiser,
            curr_loss=best_loss
        )
        evaluate.test_model(
            model=model,
            indata=test_dl,
            outfile=checkpoint_path + '/' + run_name + str(epoch) + "singleNote.wav",
        )
        # if epoch % 10 == 0:  # save an example processed audio file
        evaluate.run_file_through_model(
            model=model,
            infile=config.TEST_FILE_PATH,
            outfile=checkpoint_path + '/' + run_name + str(epoch) + ".wav",
        )
    print("epoch, train, val ", epoch, ep_loss, val_loss)
