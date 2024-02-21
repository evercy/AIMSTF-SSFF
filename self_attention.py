import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


##############################################################################################################
"""The feature dimension sent into module must be (Time, batch_size, dim)"""

""" (batch_size, Time, dim)   --->   (Time, batch_size, dim)"""
##############################################################################################################


class Attention(nn.Module):

    # Support multihead attention ,default num_heads = 1
    def __init__(self, dim, num_heads=1, attn_dropout=0., bias=True, add_bias_kv=False):
        super(Attention, self).__init__()
        self.embed_dim = dim      # d
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = dim // num_heads         # d_k     d为输入特征维度，d_k为其查询向量维度
        assert self.head_dim * num_heads == self.embed_dim,      "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5    # Scaled dot product model

        # 创建 parameter 参数
        self.in_proj_weight = Parameter(torch.Tensor(3 * dim, dim))
        self.register_parameter('in_proj_bias', None)

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * dim))

        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, dim))
        else:
            self.bias_k = self.bias_v = None

        self.reset_parameters()     # layer Norm

    # 输入、输出权重 xavier 初始化,服从均匀分布
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # 输入、输出偏置初始化为常数 0
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value):   # query is input  features
        """
        Input shape: Time x Batch x dim    Self-attention can be implemented by
        passing in the same arguments for query, key and value
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()   # 判断qkv是不是一样，返回True或False
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()       # k、v 大小一致

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        # 缩放
        q = q * self.scaling       # (len , bsz, d)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])  # (len+1, bsz, d)
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

        #   q = (bsz * num_head, len, d_k)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        #   k, v = (bsz * num_head, len+1, d_k)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))    # 缩放点积模型
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)    # y_a
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)     # Linear layer intergated

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):    # query is input features
        return self._in_proj(query).chunk(3, dim=-1)      # (len, bsz, 3d)  --> 3 * (len, bsz, d)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    """Linear transformation is used to generate query vectors and key value pairs"""
    def _in_proj(self, input, start=0, end=None, **kwargs):   # input  -->  (len, bsz, d)
        weight = kwargs.get('weight', self.in_proj_weight)    # (3d, d)
        bias = kwargs.get('bias', self.in_proj_bias)      # (3d)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)      # (len, bsz, 3d)


class QueryAttention(nn.Module):

    def __init__(self, dim, attn_dropout=0., bias=True, add_bias_kv=False):
        super(QueryAttention, self).__init__()
        """dim = d_q = d_k = d_v"""
        self.dim = dim
        self.attn_dropout = attn_dropout
        self.scaling = self.dim ** -0.5     # Scaled dot product model

        # 创建 parameter 参数
        self.in_proj_weight = Parameter(torch.Tensor(2 * dim, dim))    # Parameters with only key value pairs
        self.register_parameter('in_proj_bias', None)

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(2 * dim))

        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, dim))
        else:
            self.bias_k = self.bias_v = None

        self.reset_parameters()     # layer Norm

    # 输入、输出权重 xavier 初始化,服从均匀分布
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # 输入、输出偏置初始化为常数 0
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    """The attention model query vector is given by the model"""
    """d_q = d_k = d_v = d"""

    def forward(self, query, input_feature):     # The query vector is existing ,  x is used to generate key value pairs

        # Normally, query and input_feature are the same size
        t, b, dim = query.size()
        #t, b, c, s = query.size()
        time, bsz, embed_dim = input_feature.size()
        #time, bsz, channel, spatial = input_feature.size()

        assert embed_dim == self.dim == dim
        assert list(query.size()) == [time, bsz, embed_dim]

        # generate key value pairs
        k, v = self.in_proj_kv(input_feature)

        _, _, d = k.size()
        assert dim == d

        query = query * self.scaling    # (time, bsz, d)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])  # (time+1, bsz, d)
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

        #  query = (bsz, time, d)
        query = query.contiguous().transpose(0, 1)

        #   k, v = (bsz, time+1, d)
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.dim).transpose(0, 1)

        hidden = k.size(1)     # time + 1

        attn_weights = torch.bmm(query, k.transpose(1, 2))  # 缩放点积模型
        assert list(attn_weights.size()) == [bsz, time, hidden]

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  # y_a
        assert list(attn.size()) == [bsz, time, self.dim]

        attn = attn.transpose(0, 1).contiguous().view(time, bsz, embed_dim)

        attn = self.out_proj(attn)  # Linear layer intergated

        return attn, attn_weights     # (Time, batch_size, dim)   (batch_size, Time, dim)

    def in_proj_kv(self, input_feature):
        return self._in_proj(input_feature).chunk(2, dim=-1)     # (time, bsz, 2d)  --> 2 * (time, bsz, d)

    """Linear transformation is used to generate query vectors and key value pairs"""
    def _in_proj(self, x, start=0, end=None, **kwargs):   # x is input  feature  -->  (time, bsz, d)
        weight = kwargs.get('weight', self.in_proj_weight)    # (2d, d)
        bias = kwargs.get('bias', self.in_proj_bias)      # (2d)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(x, weight, bias)      # (time, bsz, 2d)




"""a = torch.Tensor(3, 10, 30)
attention = QueryAttention(30)
attention12 = Attention(30)
_, _ = attention12(a.transpose(0, 1), a.transpose(0, 1), a.transpose(0, 1))
b, c = attention(a.transpose(0, 1), a.transpose(0, 1))

total_params0 = sum(p.numel() for p in attention12.parameters())
total_params = sum(p.numel() for p in attention.parameters())

print(b.size(), c.size(), total_params0,total_params)"""
