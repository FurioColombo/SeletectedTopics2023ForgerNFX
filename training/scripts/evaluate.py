import random

from torch.utils.data import DataLoader
from modules.utils.file_system import load_wav_file
import soundfile
import os.path
import torch 


def run_file_through_model(model, infile, outfile, samplerate=44100):
    """
    read the sent file from disk, pass it through the model
    and back out to the other file 
    """
    indata = load_wav_file(infile, want_samplerate=samplerate)
    outputs = model.forward(torch.tensor(indata))
    outputs = outputs.cpu().detach().numpy()
    if not os.path.isdir(outfile):
        soundfile.write(outfile, outputs, samplerate)
    

def test_model(model, outfile, indata=DataLoader, samplerate=44100):
    """
    read the sent file from disk, pass it through the model
    and back out to the other file
    """
    random_sample = next(iter(indata))[0]
    outputs = model.forward(random_sample)
    outputs = outputs.cpu().detach().numpy()
    r = random.randint(0, outputs.shape[0]-1)
    outputs = outputs[r, :, 0]
    print('wav file: path: ', outfile)
    if not os.path.isdir(outfile):
        soundfile.write(outfile, outputs, samplerate)

