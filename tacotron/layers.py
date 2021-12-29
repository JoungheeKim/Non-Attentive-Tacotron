import torch
from torch import nn
import math

class Dummy(nn.Module):
    """
        Dummy function for None activation
    """

    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, x):
        return x

## Activation Function used in Other Module
ACTIVATION_GROUP = {
    'linear':Dummy,
    'tanh': nn.Tanh,
    'relu' : nn.ReLU
}

class LinearNorm(nn.Module):
    """
        linear layer with xavier initialization
        https://github.com/NVIDIA/tacotron2/blob/185cd24e046cc1304b4f8e564734d2498c6e2e6f/layers.py#L8
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """
        1d convolution layer with xavier initialization
        https://github.com/NVIDIA/tacotron2/blob/185cd24e046cc1304b4f8e564734d2498c6e2e6f/layers.py#L21
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', init_option='xavier'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class PositionalEncoding(nn.Module):
    """
        sinusoidal positional encoding
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x]