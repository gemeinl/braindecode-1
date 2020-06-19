import torch
from torch import nn
from torch.nn.utils import weight_norm

from .modules import Ensure4d, Expression
from .functions import squeeze_final_output


class TemporalConvNet(nn.Sequential):
    """Temporal Convolutional Network

    Parameters
    ----------
    num_blocks: int
        number of temporal blocks in the network
    input_size: int
        number of input channels
    output_size: int
    num_channels: int
        number of output channels of each convolution
    kernel_size: int
        size of the convolutions
    drop_prob: float
        dropout probability
    """
    def __init__(self, num_channels, num_blocks, kernel_size, drop_prob,
                 input_size, output_size):
        super().__init__()
        self.add_module("ensuredims", Ensure4d())
        temporal_blocks = nn.Sequential()
        for i in range(num_blocks):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels
            temporal_blocks.add_module(
                "temp_block_{:d}".format(i), _TemporalBlock(
                    in_channels, num_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, drop_prob=drop_prob
                ))
        self.add_module("temporal_blocks", temporal_blocks)
        self.fc = nn.Linear(
            in_features=num_channels, out_features=output_size, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.add_module("squeeze", Expression(squeeze_final_output))

        self.min_len = 1
        for i in range(num_blocks):
            dilation = 2 ** i
            self.min_len += 2 * (kernel_size - 1) * dilation

    def forward(self, x):
        # TODO: clean this up
        batch_size = x.size(0)
        time_size = x.size(2)
        # RNN format: N x L x C
        # Transpose to CNN format:  N x C x L
        # x = transpose(x, 1, 2)
        print(x.shape)
        x = x.squeeze(-1)
        print(x.shape)
        x = self.temporal_blocks(x)
        print(x.shape)
        # RNN format: N x L x C
        x = torch.transpose(x, 1, 2).contiguous()

        fc_out = self.fc(x.view(batch_size * time_size, x.size(2)))
        fc_out = self.log_softmax(fc_out)
        fc_out = fc_out.view(batch_size, time_size, fc_out.size(1))

        out_size = 1 + max(0, time_size - self.min_len)
        return fc_out[:, -out_size:, :].transpose(1, 2).contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, drop_prob):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(drop_prob)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(drop_prob)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return 'chomp_size={}'.format(self.chomp_size)

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
