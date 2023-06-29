## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
import os
import torch
from datetime import datetime
import config
import shutil
from abc import ABC


class Model(torch.nn.Module, ABC):
    def save_model(self, out_dir, model, curr_epoch, optimizer, curr_loss):
        # data for folder names
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
            model=model,
            curr_epoch=curr_epoch,
            optimizer=optimizer,
            curr_loss=curr_loss
        )

        # save fro rtneural
        outfile = new_dir_path + '/model.json'
        self._save_for_rtneural(outfile=outfile)

        print("saved model checkpoints in", new_dir_path, "folder")

        return new_dir_path
