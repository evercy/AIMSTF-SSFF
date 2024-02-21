import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from math import sqrt


# The implementation of CONVLSTM are based on the code from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, input_tensor, time=None):
        b, _, _, h, w = input_tensor.size()

        hidden_state = self.cell.init_hidden(b, h, w)

        seq_len = input_tensor.size(1)

        h, c = hidden_state
        for t in range(seq_len):
            h, c = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h, c])
        return h


class FaceFeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FaceFeatureExtractorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'FaceFeatureExtractorCNN_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        # self.load_state_dict(torch.load(path))
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
import os

class FaceFeatureExtractor(nn.Module):
    def __init__(self, feature_size=128, pretrain=True):
        super(FaceFeatureExtractor, self).__init__()
        cnn = FaceFeatureExtractorCNN()
        if pretrain:
            Path = os.path.abspath(os.path.join(os.getcwd(), ".."))
            cnn.load(Path+ '/pretrained_cnn.pth')
        self.cnn = cnn.net
        self.rnn = ConvLSTM(128, 128)
        self.fc = nn.Linear(128 * 6 * 6, feature_size)

    def forward(self, x):
        # input should be 5 dimension: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        cnn_output = self.cnn(x)

        rnn_input = cnn_output.view(b, t, 128, 6, 6)
        rnn_output = self.rnn(rnn_input)
        rnn_output = torch.flatten(rnn_output, 1)
        output = self.fc(rnn_output)
        #('output_face',output.shape)
        return output