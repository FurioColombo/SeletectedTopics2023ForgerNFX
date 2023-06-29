## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
import os
import json
from datetime import datetime
from json import JSONEncoder
import shutil
import torch

from config import config

class SimpleConv1d(torch.nn.Module):
    """
    LSTM Model after
    A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
    uses 32 hidden by default.
    Wright et al. showed decent performance for 32, but
    even better going up to 96
    """

    def __init__(self, in_channels=1, out_channels=1, stride=1, kernel_size=8, normalize_output=False):
        super().__init__()
        # Batch first means input data is [batch,sequence,feature]
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.dense = torch.nn.Linear(kernel_size, 1)  # linear before normalization
        self.tanh = torch.nn.Tanh()
        self.normalize_output = normalize_output

    def forward(self, torch_in):
        print('conv_model.py - torch in shape:', torch_in.shape)
        # int shape must be 1 dimensional of  length input length,
        # which is the number of the number of sample to feed it
        x, _ = self.conv1d(torch_in)

        dense = self.dense(x)

        if self.normalize_output:
            return self.tanh(dense)
        else:
            return dense

    # ====================================== SAVE MODEL ======================================
    def _save_for_rtneural(self, outfile):
        # used for saving .json file for torch checkpoint
        class EncodeTensor(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().detach().numpy().tolist()
                return super(json.NpEncoder, self).default(obj)

        with open(outfile, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=EncodeTensor)

    def _save_torch_model(self, outfile, curr_epoch, optimizer, curr_loss):
        # used for saving .pt file for the torch model
        torch.save({
            'epoch': curr_epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': curr_loss,
        }, outfile)

        now = datetime.now()
        date_time = now.strftime("_%m-%d-%Y_%H-%M")

        # pytorch folder path
        subfolder_path = "checkpoints/pytorch"

        # create subsections
        checkpoints_subsections = config.DATASET_TARGET_FOLDER_NAME.split('/')

        for folder in checkpoints_subsections:
            if folder != '':
                subfolder_path = subfolder_path + "/" + folder
                if not os.path.isdir(subfolder_path):
                    os.mkdir(subfolder_path)

        # create checkpoints folder
        new_dir_name = date_time
        new_dir_path = subfolder_path + "/" + new_dir_name
        if not os.path.isdir(new_dir_path):
            os.mkdir(new_dir_path)

        # copy config file
        shutil.copy('EGFx/config/config.py', new_dir_path + '/config.py')

        # save torch model
        outfile = new_dir_path + '/model.pt'
        self._save_torch_model(
            outfile=outfile,
            curr_epoch=curr_epoch,
            optimizer=optimizer,
            curr_loss=curr_loss
        )

        # save for rtneural
        outfile = new_dir_path + '/model.json'
        self._save_for_rtneural(outfile=outfile)

        print("saved model checkpoints in", new_dir_path, "folder")

        return new_dir_path
