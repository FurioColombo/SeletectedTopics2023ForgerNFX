## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
import torch
from modules.models.abstract_models import NNModel


class SimpleLSTM(NNModel):
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

