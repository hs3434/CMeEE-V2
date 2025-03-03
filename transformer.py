import torch
import math
import copy
from torch import nn
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Embeddings(nn.Module):
    def __init__(self, d_model, d_vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, d_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class PositionalEncoding(nn.Module):
    """实现Positional Encoding功能"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        位置编码器的初始化函数
        :param d_model: 词向量的维度，与输入序列的特征维度相同，512
        :param dropout: 置零比率
        :param max_len: 句子最大长度,5000
        """
        # 如果d_model不是偶数那么后面pe[:, 1::2]会少一个维度，赋值报错。
        assert d_model % 2 == 0
        super(PositionalEncoding, self).__init__()
        # 初始化一个nn.Dropout层，设置给定的dropout比例
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵
        # (5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，每个位置用一个512维度向量来表示其位置编码
        pe = torch.zeros(max_len, d_model)
        # 偶数和奇数在公式上有一个共同部分，使用log函数把次方拿下来，方便计算
        # position表示的是字词在句子中的索引，如max_len是128，那么索引就是从0，1，2，...,127
        # 论文中d_model是512，2i符号中i从0取到255，那么2i对应取值就是0,2,4...510
        # (5000) -> (5000,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算用于控制正余弦的系数，确保不同频率成分在d_model维空间内均匀分布
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 根据位置和div_term计算正弦和余弦值，分别赋值给pe的偶数列和奇数列
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # 从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # 从1开始到最后面，补长为2，其实代表的就是奇数位置
        # 上面代码获取之后得到的pe:[max_len * d_model]
        # 下面这个代码之后得到的pe形状是：[1 * max_len * d_model]
        # 多增加1维，是为了适应batch_size
        # (5000, 512) -> (1, 5000, 512)
        pe = pe.unsqueeze(0)
        # 将计算好的位置编码矩阵注册为模块缓冲区（buffer），这意味着它将成为模块的一部分并随模型保存与加载，但不会被视为模型参数参与反向传播
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        x = (
            x + self.pe[:, : x.size(1)].clone().detach()
        )  # 经过词向量的输入与位置编码相加
        # clone().detach() 组合起来的主要作用是创建一个与原张量数据相同，但不参与梯度计算且内存独立的新张量
        # Dropout层会按照设定的比例随机“丢弃”（置零）一部分位置编码与词向量相加后的元素，
        # 以此引入正则化效果，防止模型过拟合
        return self.dropout(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    query: (len_q, d_k)
    key: (len_v, d_k)
    value: (len_v, d_v)
    mask: (len_q, len_v) fill true
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        We assume d_v always equals d_k.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (len_q, d_k)
        key: (len_v, d_k)
        value: (len_v, d_k)
        mask: (len_q, len_v) fill true
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


def len_mask(batch_mask) -> torch.Tensor:
    return batch_mask.unsqueeze(-2)


def len_mask2(batch_mask, len_q) -> torch.Tensor:
    return (
        batch_mask.clone()
        .unsqueeze(-2)
        .expand(*batch_mask.shape, len_q)
        .transpose(-1, -2)
    )


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 1


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    A standard Transformer architecture.
    """

    def __init__(
        self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    ):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N
        )
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        self.generator = Generator(d_model, tgt_vocab)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def inference_test():
    test_model = Transformer(11, 11, 2)
    # test_model.eval() 评估模式，比如 Dropout 层在评估和训练时表现不同。
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    src_mask = src_mask == 0

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()
