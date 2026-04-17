import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # 初始化权重矩阵。由输入变换而来
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))
        # 初始化权重矩阵。由输入变换而来
        self.w_key = nn.Parameter(torch.rand(d_in, d_out))
        # 初始化权重矩阵。由输入变换而来
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # 自注意力机制中，计算q、k、v受输入序列的关注程度
        keys = x @ self.w_key
        queries = x @ self.w_query
        values = x @ self.w_value
        # 计算查询与键的注意力得分
        attn_scores = queries @ keys.T
        # 归一化，得到注意力权重（表征模型对输入序列的关注程度）
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # 注意力权重与值向量进行矩阵相乘（加权），得到关注程度最高的值，即，创建上下文向量
        context_vec = attn_weights @ values
        return context_vec


class SelfAttentionV2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        # 计算查询（Q）与键（Key）的注意力得分，此时这个K是输入的整个序列。
        # 即：计算查询（Q）与整个输入序列的相关性
        # 为什么是乘以 keys的转置矩阵。是为了提取行向量的特征
        attn_scores = queries @ keys.T
        # 归一化，得到注意力权重矩阵（表征模型中查询Q对输入序列的关注程度）
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # 计算注意力权重与值向量进行矩阵相乘（加权求和，提取真正的内容），得到上下文向量
        context_vec = attn_weights @ values
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # 初始化q、k、v权重
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 添加一个dropout层
        self.dropout = nn.Dropout(dropout)
        # register_buffer调用也是一个新版本
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        # 将维度 1 和 2 转置，将批维度保持在第一个位置（0）
        attn_scores = queries @ keys.transpose(1,2)
        # 在 PyTorch 中，带有尾随下划线的操作将就地执行，从而避免了不必要的内存副本
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens],
            -torch.inf
        )
        # 归一化处理
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5,dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    """
    叠加多个单头注意力层
    """
    def __init__(self, d_in, d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                # 单头注意力层
                CausalAttention(d_in,d_out,context_length,dropout,qkv_bias) for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # 减少维度，以匹配所需的输出维度
        self.head_dim = d_out // num_heads
        # 初始化权重
        self.w_query=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value=nn.Linear(d_in, d_out, bias=qkv_bias)
        # 使用一个线性层来组合头的输出
        self.out_proj = nn.Linear(d_out, d_out)
        # dropout。避免过拟合
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            # 掩码权重
            torch.triu(
                torch.ones(context_length,context_length),
                diagonal=1
            )
        )

    def forward(self,x):
        b,num_tokens,d_in = x.shape
        # 计算键矩阵
        keys = self.w_key(x)
        # 计算查询矩阵
        queries = self.w_query(x)
        # 计算值矩阵
        values = self.w_value(x)
        # .view() 张量重塑
        # 通过添加一个num_heads维度来隐式地分隔矩阵。
        # 然后展开最后一个维度：(b, num_tokens,d_out) -> (b, num_tokens,self.num_heads,self.head_dim)
        keys = keys.view(b, num_tokens,self.num_heads,self.head_dim)
        values = values.view(b, num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b, num_tokens,self.num_heads,self.head_dim)
        # 转换矩阵。从形状(b, num_tokens,self.num_heads,self.head_dim)
        # 转换到(b, self.num_heads,num_tokens,self.head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        # 计算查询（queries）与模型中输入序列（keys）的相关性。计算每个头的点积
        attn_scores = queries @ keys.transpose(2,3)
        # 截断词元数量的掩码
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        # 使用掩码来填充注意力分数。使模型不关心未来的词元
        attn_scores.masked_fill_(mask_bool,-torch.inf)
        # 计算注意力权重
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5,
            dim=-1
        )
        # dropout。防止过拟合
        attn_weights = self.dropout(attn_weights)
        # 计算上下文向量。即：使用注意力权重与值向量（V）矩阵相乘，提取相关的真实内容。
        # 张量形状：(b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)
        # 组合头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b,num_tokens,self.d_out
        )
        # 添加一个可选的线性投影。这个操作融合多头注意力。out_proj是一个全连接层，数学上，全连接层本质是加权求和，
        # 通过加权求和将不同注意力头的信息进行线性组合（多头注意力汇总）。
        # 通过线性层，让不同头的信息发生“化学反应”。
        # 输入是 d_out，输出也是 d_out，但内部特征已经被重新加权混合了
        context_vec = self.out_proj(context_vec)
        return context_vec





if __name__ == "__main__":
    # 输入词元的向量表示
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )
    inputs1 = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts (x^3)
         [0.22, 0.58, 0.33],  # with (x^4)
         [0.77, 0.25, 0.10],  # one (x^5)
         [0.05, 0.80, 0.55]]  # step (x^6)
    )

    # torch.manual_seed(123)
    # self_attention_v1 = SelfAttentionV1(3, 6)
    # print(self_attention_v1(inputs))

    # torch.manual_seed(789)
    # sa_v2 = SelfAttentionV2(3, 6)
    # print(sa_v2(inputs))

    # torch.manual_seed(123)
    # # 选择使用50%的dropout率
    # dropout = torch.nn.Dropout(0.5)
    # # 创建一个全1矩阵
    # example = torch.ones(6,6)
    # print(dropout(example))

    # torch.manual_seed(123)
    # context_length = inputs.shape[1]
    # ca = CausalAttention(3,6,context_length,0.0)
    # context_vec = ca(inputs)
    # print(f"context_vec: {context_vec}")


    input_inputs = [inputs, inputs1]
    # 堆叠两个向量
    batch = torch.stack(input_inputs,dim=0)
    print(batch)




