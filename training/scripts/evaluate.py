import random

import numpy as np
from torch.utils.data import DataLoader

import config
from training.modules.utils.file_system import load_wav_file
import soundfile
import os.path
import torch 


def run_file_through_model(model, infile, out_dir, file_name ,samplerate=44100):
    """
    read the sent file from disk, pass it through the model
    and back out to the other file 
    """
    audio_out_folder_path = os.path.join(out_dir, 'audio')
    if not os.path.isdir(audio_out_folder_path):
        os.mkdir(audio_out_folder_path)
    out_file_path = os.path.join(audio_out_folder_path, file_name)

    # load file for testing
    indata = load_wav_file(infile, target_samplerate=samplerate)

    # reshape file according to training parameters
    n_samples_to_trim = indata.shape[0] % config.NN_IN_BLOCK_SIZE
    input_shape = (int(indata.shape[0] / config.NN_IN_BLOCK_SIZE), config.NN_IN_BLOCK_SIZE)
    indata = np.reshape(
        indata[:-n_samples_to_trim or None, :],
        input_shape
    )

    # run file through the model
    outputs = model.forward(
        torch.tensor(indata)
    )

    outputs = outputs.cpu().detach().numpy().flatten()

    if not os.path.isdir(out_file_path):
        soundfile.write(out_file_path, outputs, samplerate)
    

def test_model(model, out_dir, file_name, indata=DataLoader, samplerate=44100):
    """
    read the sent file from disk, pass it through the model
    and back out to the other file path
    """
    audio_out_folder_path = os.path.join(out_dir, 'audio')
    if not os.path.isdir(audio_out_folder_path):
        os.mkdir(audio_out_folder_path)
    out_file_path = os.path.join(audio_out_folder_path, file_name)

    random_sample = next(iter(indata))[0]
    outputs = model.forward(random_sample)
    outputs = outputs.cpu().detach().numpy()
    r = random.randint(0, outputs.shape[0]-1)
    outputs = np.reshape(outputs, newshape=(outputs.shape[0], -1))[r, :]
    print('wav file: path: ', out_file_path)
    if not os.path.isdir(out_file_path):
        soundfile.write(out_file_path, outputs, samplerate)

