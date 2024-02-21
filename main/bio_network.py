import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
from self_attention import Attention, QueryAttention
from torch.nn import init
B_INIT= -0.2
class SE_Block(nn.Module):
    def __init__(self, in_chs, reduction=4):
        super(SE_Block, self).__init__()
        #self.gate_fn = gate_fn
        reduced_chs = in_chs // reduction
        #reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_reduce = nn.Conv1d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv1d(reduced_chs, in_chs, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.act2(x_se)
        return x


class baseconv(nn.Module):

    def __init__(self, eeg_channel=32, s_channel=8):
        # input_size:  EEG channel x datapoint
        super(baseconv, self).__init__()
        self.s_convdown = nn.Sequential(
            nn.Conv1d(eeg_channel, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            nn.Conv1d(24, s_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(s_channel),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        #print('x',x.shape)
        out = self.s_convdown(x)

        return out


def conv_block(in_chan, out_chan, kernel, step, pool):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_chan, out_channels=out_chan,
                  kernel_size=kernel, stride=step),
        nn.LeakyReLU(),
        nn.AvgPool1d(kernel_size=pool, stride=pool))


class s_MultiScaleConv(nn.Module):

    def get_out_ch(self):
        return self.SS_channel

    def __init__(self, sampling_rate, s_channel):
        # input_size: 1 x EEG channel x datapoint
        super(s_MultiScaleConv, self).__init__()
        self.pool = 1
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.s_channel = 0
        self.Sception1 = nn.Sequential(
            nn.Conv1d(in_channels=sampling_rate, out_channels=sampling_rate,kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.Sception2 =  nn.Sequential(
        nn.Conv1d(in_channels=sampling_rate, out_channels=sampling_rate,kernel_size=3, stride=1,padding=1),
        nn.LeakyReLU()
        )
        self.BN_t = nn.BatchNorm1d(sampling_rate)
        self.SS_channel = 2*s_channel

    def forward(self, x):
        x_s = x.permute(0, 2, 1)
        y = self.Sception1(x_s)
        out = y
        y = self.Sception2(x_s)
        out = torch.cat((out, y), dim=2)
        out = self.BN_t(out)
        out = out.permute(0, 2, 1)
        return out


class t_MultiScaleConv(nn.Module):

    def get_out_ch(self):
        return self.t_channel

    def __init__(self, sampling_rate, s_channel):
        # input_size: 1 x EEG channel x datapoint
        super(t_MultiScaleConv, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 2
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.t_channel = 0
        self.Tception1 = conv_block(s_channel, s_channel, int(self.inception_window[0] * sampling_rate), 1, self.pool)
        self.Tception2 = conv_block(s_channel, s_channel, int(self.inception_window[1] * sampling_rate), 1, self.pool)
        self.Tception3 = conv_block(s_channel, s_channel, int(self.inception_window[2] * sampling_rate), 1, self.pool)
        self.BN_t = nn.BatchNorm1d(s_channel)
        for i in range(len(self.inception_window)):
            f_kernel = int(self.inception_window[i] * sampling_rate)
            f_length = (sampling_rate - f_kernel + 1) // self.pool
            self.t_channel += f_length

    def forward(self, x):
        y = self.Tception1(x)
        #print('y',y.shape)
        out = y
        y = self.Tception2(x)
        #print('y', y.shape)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        #print(out.shape)
        return out


class TSAM(nn.Module):
    def __init__(self,  s_channel, t_channel, s_kernel, t_kernel):
        super(TSAM, self).__init__()
        self.s_kernel = s_kernel
        self.t_kernel = t_kernel
        self.t_pool = 1

        self.S_Attention = nn.Sequential(
            nn.Conv1d(t_channel, t_channel, kernel_size=self.t_kernel, padding=1, stride=1),
            nn.BatchNorm1d(t_channel),
            nn.LeakyReLU(),
        )
        self.se = SE_Block(s_channel)

        self.T_Attention = nn.Sequential(
            nn.Conv1d(s_channel, s_channel, kernel_size=self.s_kernel, padding=1, stride=1),
            nn.BatchNorm1d(s_channel),
            nn.LeakyReLU(),
        )
        self.assist_attention = QueryAttention(s_channel, attn_dropout=0.2)
        self.attention = Attention(t_channel, attn_dropout=0.2, num_heads=4)

    def forward(self, x):
        #print('x', x.shape)
        x_t = self.T_Attention(x)
        x_t_att, _ = self.attention(x_t, x_t, x_t)
        #print('x_t_att', x_t_att.shape)

        x_s = x.permute(0, 2, 1)
        x_s = self.S_Attention(x_s)
        x_s_se = x_s.permute(0, 2, 1)
        x_s_se = self.se(x_s_se)
        #print('x_s_se', x_s_se.shape)
        tasm_out = x_t_att + x_s_se
        #print('tasm_out', tasm_out.shape)
        return tasm_out


class GateGenerate(nn.Module):
    def __init__(self, s_channel, t_channel, hidden_dim, kernel_size, bias, dtype):
        super(GateGenerate, self).__init__()
        input_size = (s_channel, t_channel)
        self.S_EEG, self.T_EEG = input_size
        #self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.data_type = dtype

        self.conv_gates = nn.Conv1d(in_channels=2*s_channel,
                                    out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=1,
                                    bias=self.bias)
        self.h_cur_conv = nn.Sequential(nn.Conv1d(in_channels=s_channel,out_channels=self.hidden_dim,kernel_size=kernel_size,padding=1),
                                        nn.BatchNorm1d(self.hidden_dim),
                                        nn.LeakyReLU(),)
        self.conv_can = nn.Conv1d(in_channels=s_channel,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=self.bias)
        self.out_cur_conv = nn.Sequential(nn.Conv1d(in_channels=self.hidden_dim, out_channels=s_channel, kernel_size=kernel_size, padding=1),
                                          nn.BatchNorm1d(s_channel),
                                          nn.LeakyReLU(), )
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_dim, self.S_EEG, self.T_EEG)).type(self.data_type)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        #print('combined_conv', combined_conv.shape)
        h_cur = self.h_cur_conv(h_cur)
        #print('h_cur', h_cur.shape)
        update_gate = torch.sigmoid(combined_conv)
        cc_cnm = self.conv_can(input_tensor)
        #print('cc_cnm', cc_cnm.shape)
        cnm = torch.tanh(cc_cnm)
        h_next = update_gate * h_cur + (1 - update_gate) * cnm
        #print('h_next', h_next.shape)
        h_next = self.out_cur_conv(h_next)
        #print('h_next', h_next.shape)
        return h_next


def cosine_similarity(x, y):
    # 将 x 和 y 的形状调整为 (B, S, T)
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)

    similarity = torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=1)
    return similarity


class DualSpatialTimeModel(nn.Module):
    def __init__(self, sampling_rate, eeg_channel, s_channel,
                 s_kernel, t_kernel, t_channel,
                 hidden_dim, kernel_size, bias, dtype,
                 num_layers, eeg_feature_size):
        super(DualSpatialTimeModel, self).__init__()

        # self.s_MultiScaleConv = s_MultiScaleConv(sampling_rate, eeg_channel)
        # ss_channel = self.s_MultiScaleConv.get_out_ch()
        # s_channel = ss_channel//2
        self.t_MultiScaleConv = t_MultiScaleConv(sampling_rate, s_channel)
        t_channel = self.t_MultiScaleConv.get_out_ch()
        #self.baseconv = baseconv(eeg_channel, s_channel)
        self.num_layers = num_layers

        self.TASM_layers = nn.ModuleList([TSAM(s_channel, t_channel, s_kernel, t_kernel)
                                          for _ in range(self.num_layers)])
        #self.gate_layers = nn.ModuleList([GateGenerate(s_channel, s_channel) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([GateGenerate(s_channel, t_channel, hidden_dim,
                                                       kernel_size, bias, dtype) for _ in range(num_layers)])
        # self.conv_down = nn.Conv1d(186, 36, 3, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(s_channel*t_channel, eeg_feature_size),
        )

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def forward(self, x):

        current_features = []
        previous_features = []
        #s_MultiScaleConv_out = self.s_MultiScaleConv(x)
        t_MultiScaleConv_out = self.t_MultiScaleConv(x)
        #base_x = self.baseconv(x)
        previous_features.append(t_MultiScaleConv_out)

        # weighted_features = []
        for i in range(1, self.num_layers + 1):
            #print('multi_scale_out', previous_features[i - 1].shape)
            TASM_output = self.TASM_layers[i - 1](previous_features[i - 1])
            current_features.append(TASM_output)

            # current_feature = features[i]
            # 计算当前特征与先前特征的相似度
            similarities = []

            for j in range(i):
                previous_feature_j = previous_features[j]
                similarity = cosine_similarity(current_features[i-1], previous_feature_j)
                similarities.append(similarity)
            similarities = torch.stack(similarities, dim=1)
            # 根据相似度计算加权系数
            if i == 1:
                weights = similarities
            else:
                weights = F.softmax(similarities, dim=1)
            #print('weights',weights)
            weights = 1-weights
            # 将各层次特征与相应的加权系数进行加权
            weighted_feature = torch.sum(
                weights.unsqueeze(-1).unsqueeze(-1) * torch.stack(previous_features[:i], dim=1), dim=1)
            #print('weights', weights)
            # TASM_output = TASM_output.permute(2, 0, 1)
            # weighted_feature = weighted_feature.permute(2, 0, 1)
            gate_output = self.gate_layers[i - 1](TASM_output, weighted_feature)
            # print('gate_output', gate_output.shape)
            # print('current_features[i-1]', current_features[i-1].shape)
            gate_output = gate_output + current_features[i-1]
            #gate_output = gate_output.permute(1, 2, 0)

            previous_features.append(gate_output)

        final_features = previous_features[self.num_layers]
        final_features = torch.as_tensor(final_features, dtype=torch.float32)
        #print('final_features', final_features.shape)
        fin_out  = torch.flatten(final_features, 1)
        eeg_out = self.fc(fin_out)
        #print('eeg_out', eeg_out.shape)

        return eeg_out

