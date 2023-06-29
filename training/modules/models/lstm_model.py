## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
import os
from json import JSONEncoder
import torch
import json
from datetime import datetime
from config import config
import shutil


class SimpleLSTM(torch.nn.Module):
    """
    LSTM Model after
    A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
    uses 32 hidden by default.
    Wright et al. showed decent performance for 32, but 
    even better going up to 96
    """

    def __init__(self, hidden_size=32, normalize_out=False):
        super().__init__()
        # Batch first means input data is [batch,sequence,feature]
        self.lstm = torch.nn.LSTM(1, hidden_size, batch_first=True)
        self.dense = torch.nn.Linear(hidden_size, 1)  # from  hidden back to 1 output
        self.tanh = torch.nn.Tanh()
        self.drop_hidden = True
        self.data01normalized = normalize_out

    def zero_on_next_forward(self):
        """
        next time forward is called, the network will
        run it with zeroed hidden+cell values
        """
        self.drop_hidden = True

    def forward(self, torch_in):
        if self.drop_hidden:
            batch_size = torch_in.shape[0]
            # h_size = (num_layers, batch_size, hidden_count)
            h_shape = [self.lstm.num_layers, batch_size, self.lstm.hidden_size]
            # print("dropping hidden, shape probably ", h_shape)
            hidden = torch.zeros(h_shape).to(torch_in.device)
            cell = torch.zeros(h_shape).to(torch_in.device)
            # print('lstm_models.py - torch in shape:', torch_in.shape)
            x, _ = self.lstm(torch_in, (hidden, cell))
            self.drop_hidden = False
        else:
            x, _ = self.lstm(torch_in)
        dense = self.dense(x)

        if self.data01normalized:
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
        curr_epoch_dir_name = config.DATASET_TARGET_FOLDER_NAME.replace('/', '') + '_' + config.MODEL_NAME
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
        torch_checkpoints_folder_path = os.path.join(curr_epoch_dir_path, 'pytorch')   # /checkpoints/pytorch
        if not os.path.isdir(torch_checkpoints_folder_path):
            os.mkdir(torch_checkpoints_folder_path)

        # rtneural
        rtneural_checkpoints_folder_path = os.path.join(curr_epoch_dir_path, 'rtneural')   # /checkpoints/rtneural
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
