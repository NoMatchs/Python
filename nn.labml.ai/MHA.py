import math
from typing import Optional, List
import torch
from torch import nn
from torch import Tensor


# ==========================================
# 1. 多头注意力的线性变换准备
# ==========================================
class PrepareForMultiHeadAttention(nn.Module):
    """
    这部分执行线性变换，并将向量分割成给定数量的头以获得多头注意力。
    适用于键、查询和值向量。
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # 线性层用于线性变换
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # 注意力头数
        self.heads = heads
        # 每个头部中向量的维度数量
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # 输入的形状为 [seq_len, batch_size, d_model] 或 [batch_size, d_model]
        # 我们对最后一维应用线性变换，并将其分为多个头。
        # 线性变换
        x = self.linear(x)
        # 将最后一个维度分成多个头部
        head_shape = x.shape[:-1]
        x = x.view(*head_shape, self.heads, self.d_k)
        # 输出具有形状 [seq_len, batch_size, heads, d_k] 或 [batch_size, heads, d_model]
        return x


# ==========================================
# 2. 多头注意力模块
# ==========================================
class MultiHeadAttention(nn.Module):
    """
    这计算给出对于 key、value 和 query 向量缩放后的多头注意力。
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(
        self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True
    ):
        super().__init__()

        # 每个头部的特征数量
        self.d_k = d_model // heads
        # 注意力头数
        self.heads = heads

        # 这些将对多头注意力的 query、key 和 value 进行转换。
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # 在键(Key)的时间维度上进行注意力 Softmax
        self.softmax = nn.Softmax(dim=1)
        # 输出层
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Softmax 之前的缩放系数
        self.scale = 1 / math.sqrt(self.d_k)

        # 存储注意力信息，以便在需要时用于记录或其他计算。
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        计算 Query 和 Key 之间的分数。
        计算方法可以同样适用于其他整体，如相对注意力。
        计算 QK^T 或 S_ij = sum_q Q_iq * K_jq
        """
        return torch.einsum("ibhd,jbhd->ijbh", query, key)

    def prepare_mask(
        self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]
    ):
        """
        mask 的形状为 [seq_len_q, seq_len_k, batch_size]，其中第一维是查询维度。
        如果查询维度等于 1，则会进行广播。
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # 所有的头部使用相同的掩码。
        mask = mask.unsqueeze(-1)
        # 生成的掩码形状为 [seq_len_q, seq_len_k, batch_size, heads]
        return mask

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        query, key 和 value 是存储查询、键和值向量集合的张量。
        它们的形状为 [seq_len, batch_size, d_model]。
        mask 的形状为 [seq_len, seq_len, batch_size], mask[i, j, b] 表示批次 b 中，
        在位置 i 处查询是否有权访问位置 j 处的键值对。
        """
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # 为注意力计算准备向量 query, key 并 value 它们的形状将变为
        # [seq_len, batch_size, heads, d_k]。
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # 计算注意力分数 QK^T，这将得到一个形状为
        # [seq_len, seq_len, batch_size, heads] 的张量。
        scores = self.get_scores(query, key)

        # 缩放分数 QK^T / sqrt(d_k)
        scores *= self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 对 Key 序列维度上的注意力进行 softmax 操作
        attn = self.softmax(scores)

        # 调试时保存注意力信息
        # tracker.debug('attn', attn)

        # 应用 Dropout
        attn = self.dropout(attn)

        # 乘以数值： softmax(QK^T / sqrt(d_k)) V
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # 为其他计算保存注意力信息
        self.attn = attn.detach()

        # 连接多个头
        x = x.reshape(seq_len, batch_size, -1)

        # 输出层
        return self.output(x)


# ==========================================
# 3. 测试代码 (测试模型)
# ==========================================
if __name__ == "__main__":
    # 定义超参数
    seq_len = 10
    batch_size = 2
    d_model = 512
    heads = 8
    dropout = 0.1

    # 实例化模型
    mha = MultiHeadAttention(heads=heads, d_model=d_model, dropout_prob=dropout)

    # 构造随机输入张量: [seq_len, batch_size, d_model]
    x = torch.randn(seq_len, batch_size, d_model)

    # 构造掩码 (可选)，此处模拟一个全 1 的掩码 (全部可见)
    # 形状需满足: [seq_len, seq_len, batch_size]
    mask = torch.ones(seq_len, seq_len, batch_size)

    # 前向传播
    output = mha(query=x, key=x, value=x, mask=mask)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力矩阵形状: {mha.attn.shape}")  # [seq_len, seq_len, batch_size, heads]

    # 验证输出形状是否与输入一致 (d_model)
    assert output.shape == (seq_len, batch_size, d_model)
    print("模型前向传播测试通过！")
