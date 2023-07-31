## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
import os
from datetime import datetime
import json
from json.encoder import JSONEncoder

import torch
from training.config import config
from abc import ABC, abstractmethod


class NNModel(torch.nn.Module, ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, torch_in):
        pass

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


    def save_model(self, out_dir, curr_epoch, optimizer, curr_loss):
        # data for folder names
        now = datetime.now()
        date_time = now.strftime("_%m-%d-%Y_%H-%M")

        # create paths variables and folders

        # checkpoints folder
        checkpoints_folder_path = os.path.join(out_dir, 'checkpoints')  # resources/checkpoints
        if not os.path.isdir(checkpoints_folder_path):
            os.mkdir(checkpoints_folder_path)

        # create checkpoints folder
        curr_epoch_dir_name = config.DATASET_TARGET_FOLDER_NAME.replace('/', '') + '_' + config.DATASET_TYPE + '_' \
                              + 'bl' + str(config.NN_IN_BLOCK_SIZE) + config.MODEL_NAME
        if config.MODEL_NAME == 'lstm':
            curr_epoch_dir_name = curr_epoch_dir_name + str(config.LSTM_HIDDEN_SIZE) + '_ep' + str(curr_epoch)
        elif config.MODEL_NAME == 'conv:':
            curr_epoch_dir_name = curr_epoch_dir_name + str(config.KERNEL_SIZE) + '_ep' + str(curr_epoch)
        else:
            assert 'ERROR: config model name not recognised'
        curr_epoch_dir_path = os.path.join(checkpoints_folder_path, curr_epoch_dir_name)
        if not os.path.isdir(curr_epoch_dir_path):
            os.mkdir(curr_epoch_dir_path)

        # pytorch
        torch_checkpoints_folder_path = os.path.join(curr_epoch_dir_path, 'pytorch')  # /checkpoints/pytorch
        if not os.path.isdir(torch_checkpoints_folder_path):
            os.mkdir(torch_checkpoints_folder_path)

        # rtneural
        rtneural_checkpoints_folder_path = os.path.join(curr_epoch_dir_path, 'rtneural')  # /checkpoints/rtneural
        if not os.path.isdir(rtneural_checkpoints_folder_path):
            os.mkdir(rtneural_checkpoints_folder_path)

        # save torch model
        outfile = os.path.join(torch_checkpoints_folder_path, 'model.pt')
        self._save_torch_model(
            outfile=outfile,
            curr_epoch=curr_epoch,
            optimizer=optimizer,
            curr_loss=curr_loss
        )

        # save for RTNeural
        outfile = os.path.join(rtneural_checkpoints_folder_path, 'model.json')
        self._save_for_rtneural(outfile=outfile)

        print("saved model checkpoints in", curr_epoch_dir_path, "folder")

        return curr_epoch_dir_path

