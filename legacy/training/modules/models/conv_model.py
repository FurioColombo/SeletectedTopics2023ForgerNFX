## A. Wright, E.-P. Damskägg, and V. Välimäki, ‘Real-time black-box modelling with recurrent neural networks’, in 22nd international conference on digital audio effects (DAFx-19), 2019, pp. 1–8.
import torch
from training.modules.models.abstract_models import NNModel


class SimpleConv1d(NNModel):
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
