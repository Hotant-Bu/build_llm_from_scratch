import torch
import torch.nn as nn
import tiktoken


# from src.pytorch_stu.build_llm_from_scratch.self_attention_v1 import MultiHeadAttention


class MultiHeadAttention(nn.Module):
    """
    多头注意力实现
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # 减少维度，以匹配所需的输出维度
        self.head_dim = d_out // num_heads
        # 初始化权重
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 使用一个线性层来组合头的输出
        self.out_proj = nn.Linear(d_out, d_out)
        # dropout。避免过拟合
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            # 掩码权重
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 计算键矩阵
        keys = self.w_key(x)
        # 计算查询矩阵
        queries = self.w_query(x)
        # 计算值矩阵
        values = self.w_value(x)
        # .view() 张量重塑
        # 通过添加一个num_heads维度来隐式地分隔矩阵。
        # 然后展开最后一个维度：(b, num_tokens,d_out) -> (b, num_tokens,self.num_heads,self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 转换矩阵。从形状(b, num_tokens,self.num_heads,self.head_dim)
        # 转换到(b, self.num_heads,num_tokens,self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 计算查询（queries）与模型中输入序列（keys）的相关性。计算每个头的点积
        attn_scores = queries @ keys.transpose(2, 3)
        # 截断词元数量的掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 使用掩码来填充注意力分数。使模型不关心未来的词元
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 计算注意力权重
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )
        # dropout。防止过拟合
        attn_weights = self.dropout(attn_weights)
        # 计算上下文向量。即：使用注意力权重与值向量（V）矩阵相乘，提取相关的真实内容。
        # 张量形状：(b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 组合头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        # 添加一个可选的线性投影。这个操作融合多头注意力。out_proj是一个全连接层，数学上，全连接层本质是加权求和，
        # 通过加权求和将不同注意力头的信息进行线性组合（多头注意力汇总）。
        # 通过线性层，让不同头的信息发生“化学反应”。
        # 输入是 d_out，输出也是 d_out，但内部特征已经被重新加权混合了
        context_vec = self.out_proj(context_vec)
        return context_vec


GPT_CONFIG_124M = {
    # 词汇表大小。表示会被 BPE 分词器使用的由 50 257 个单词组成的词汇表
    "vocab_size": 50257,
    # 上下文长度。指的是模型通过位置嵌入能够处理的最大输入词元数量
    "context_length": 1024,
    # 嵌入维度。可以将每个词元转化为 768 维的向量
    "emb_dim": 768,
    # 注意力头的数量。多头注意力机制中注意力头的数量
    "n_heads": 12,
    # 层数。表示模型中的 Transformer 块数量
    "n_layers": 12,
    # dropout率。表示 dropout 机制的强度（0.1 表示有 10%的隐藏单元被随机丢弃），以防止过拟合
    "drop_rate": 0.1,
    # 查询-键-值偏置。指的是是否在多头注意力机制的线性层中添加一个偏置向量，用于查询、键和值的计算
    "qkv_bias": False
}


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token嵌入向量。权重矩阵
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入向量。权重矩阵
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # dropout。防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 组合多注意力块（头）
        self.trf_blocks = nn.Sequential(
            *[
                DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])
            ]
        )
        # 归一化处理
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 输出，添加一个线性投影，融合多头注意力。即：使用全连接层进行加权求和，使关注到的不同特征进行融合
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 词元token化向量（词元嵌入向量）
        tok_embeds = self.tok_emb(in_idx)
        # 位置嵌入向量
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        # 将位置向量和token向量融合。结果使词元向量带有词元位置信息
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        # 组合多注意力块（头）
        x = self.trf_blocks(x)
        # 归一化
        x = self.final_norm(x)
        # 输出，添加一个线性投影
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    """
    层归一化
    """

    def __init__(self, emb_dim):
        super().__init__()
        # 小常数
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # 均值
        mean = x.mean(dim=-1, keepdim=True)
        # 方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化操作。减去均值，并将结果除以方差的平方根（标准差）。
        # 这里加了个小常数（self.eps）是为了防止方差为0的情况
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    GELU激活函数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
                1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        )
        )


class FeedForward(nn.Module):
    """
    前馈神经网络模块
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    """
    带有快捷链接的神经网络
    """

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # 判断是否快捷链接
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    """
    在模型的反向传播过程中计算梯度的函数
    :param model:
    :param x:
    :return:
    """
    # 前向传播
    output = model(x)
    # 目标
    target = torch.tensor([[0.]])
    # 实例化损失函数
    loss = nn.MSELoss()
    # 基于目标和模型输出之间的差距来计算损失
    loss = loss(output, target)
    # 反向传播来计算梯度
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


class TransformerBlock(nn.Module):
    """
    实现GPT中的Transformer块组件
    层归一化（LayerNorm）应用于这两个组件之前，
    而 dropout 应用于这两个组件之后，以便对模型进行正则化并防止过拟合
    """

    def __init__(self, cfg):
        super().__init__()
        # 多头注意力
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        带有快捷链接
        :param x:
        :return:
        """
        # 在注意力快中添加快捷链接
        shortcut = x
        # 层归一化
        x = self.norm1(x)
        # 计算多头注意力
        x = self.att(x)
        # dropout。防止过拟合
        x = self.drop_shortcut(x)
        # 添加原始输入到输出（即，将原始输入快捷链接到输出）
        x = x + shortcut

        # 在前馈层中添加快捷链接。
        # 即，将上一步的输出（加上了快捷链接）赋值给下一步前馈网络的快捷链接
        shortcut = x
        x = self.norm2(x)
        # 前馈神经网络计算
        x = self.ff(x)
        # dropout，防止过拟合
        x = self.drop_shortcut(x)
        # 将原始输入添加回来
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token嵌入矩阵
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入矩阵
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # dropout，防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 层归一化处理。将输出标准化，以稳定学习过程
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 最后输出经过一个线性层（线性投影），融合所有矩阵信息。升维
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        前向传播
        :param in_idx: 输入词元（词元ID）
        :return:
        """
        batch_size, seq_len = in_idx.shape
        # 将输入词元ID转换为token嵌入矩阵
        token_embeds = self.token_emb(in_idx)
        # 生成词元位置嵌入矩阵。
        # device 的设置允许我们在CPU或GPU上训练模型，具体取决于输入数据所在的设备
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 融合词元嵌入和位置嵌入。作为神经网络输入
        x = token_embeds + pos_embeds
        # dropout，防止过拟合
        x = self.drop_emb(x)
        # Transformer块计算
        x = self.trf_blocks(x)
        # 最终归一化处理（用层归一化处理）。将Transformer块的输出标准化，以稳定学习过程
        x = self.final_norm(x)
        # 输出，经过一个线性层投影，融合所有矩阵信息，升维。
        # 将 Transformer 的输出投射到分词器的词汇空间，为词汇中的每个词元生成分数（logits）
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size,
                         temperature=0.0,top_k=None, eos_id=None):
    """
    利用模型预测并生成文本
    如果大语言模型仅支持5个词元，但此时文本长度为10，则只有最后 5 个词元会被用作输入文本
    :param model:
    :param idx: idx 是当前文本的索引数组，其形状为(batch, n_tokens)
    :param max_new_tokens:
    :param context_size:
    :return:
    """

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        # 屏蔽模型参数的梯度跟踪，因为还没开始训练
        with torch.no_grad():
            logits = model(idx_cond)
        # 只关注最后一个输出的内容，因此形状会从(batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]

        # top_k采样筛选logits
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            # 使用 PyTorch 的 where 函数将低于我们选择的前 3 个词元中最低 logits 值的
            # 词元的 logits 值设置为负无穷（-inf）
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # 二项式采样
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1,keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next),dim=1)
    return idx

    #     # probas 的形状为(batch, vocab_size)
    #     probas = torch.softmax(logits, dim=-1)
    #     # idx_next 的形状为(batch, 1)
    #     idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    #     # 采用温度缩放的文本生成策略
    #     # sample = torch.multinomial(probas,num_samples=1).item()
    #     # idx_next = torch.bincount(torch.tensor(sample))
    #     # 将计算出的下一个字符的索引添加到索引数组中，
    #     # 此时 idx 的形状会变为(batch, n_tokens+1)
    #     idx = torch.cat((idx, idx_next), dim=1)
    # return idx


if __name__ == "__main__":
    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # # 编码txt1，输出词元（ID）的嵌入矩阵
    # encode_token = tokenizer.encode(txt1)
    #
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # # 融合输入token矩阵
    # batch = torch.stack(batch, dim=0)
    #
    # torch.manual_seed(123)
    # model = DummyGPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
    # print(logits)

    # ln = LayerNorm(emb_dim=5)

    # # 绘制gelu 和 relu。直观地比较 GELU 函数与 ReLU 函数
    # import matplotlib.pyplot as plt
    # gelu, relu = GELU(), nn.ReLU()
    # x = torch.linspace(-3,3,100)
    # y_gelu, y_relu = gelu(x), relu(x)
    # plt.figure(figsize=(8,3))
    # for i, (y,label) in enumerate(zip([y_gelu,y_relu],["GELU","ReLU"]),1):
    #     plt.subplot(1,2,i)
    #     plt.plot(x,y)
    #     plt.title(f"{label} activation function")
    #     plt.xlabel("x")
    #     plt.ylabel(f"{label}(x)")
    #     plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # ffn = FeedForward(GPT_CONFIG_124M)
    # x = torch.rand(2,3,768)
    # out = ffn(x)
    # print(out.shape)
    # print(out)

    # # 不包含跳跃链接（快捷链接）。观察梯度消失的效果
    # layer_sizes = [3, 3, 3, 3, 3, 1]
    # sample_input = torch.tensor([[1., 0., -1.]])
    # torch.manual_seed(123)
    # model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes=layer_sizes,use_shortcut=False)
    # # 计算损失
    # print_gradients(model_with_shortcut,sample_input)
    # # 包含一个快捷链接，观察梯度变化的效果
    # torch.manual_seed(123)
    # model_with_shortcut_link = ExampleDeepNeuralNetwork(layer_sizes=layer_sizes,use_shortcut=True)
    # print_gradients(model_with_shortcut_link,sample_input)

    # torch.manual_seed(123)
    # # 创建形状为[batch_size, num_tokens,emb_dim]的样例输入
    # x = torch.rand(2,4,768)
    # # print(x)
    # block = TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")

    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # # 编码txt1，输出词元（ID）的嵌入矩阵
    # # encode_token = tokenizer.encode(txt1)
    #
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # # 融合输入token矩阵
    # batch = torch.stack(batch, dim=0)
    # torch.manual_seed(123)
    # model = GPTModel(GPT_CONFIG_124M)
    # out = model(batch)
    # # 统计模型参数张量的总参数量。这里的测试是：163009536
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")
    # print(f"Input batch: {batch}")
    # print(f"Output batch: {out.shape}")
    # print(out)

    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print(f"encoded: {encoded}")
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"encoded_tensor.shape: {encoded_tensor.shape}")

    model = GPTModel(GPT_CONFIG_124M)
    # 禁用dropout
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print(f"Output: ", out)
    print(f"Output length: {len(out[0])}")

    # 解码预测出来的token
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"decoded_text: {decoded_text}")





