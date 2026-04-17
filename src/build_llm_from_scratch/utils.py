import re

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleTokenizerV1:
    def __init__(self, vocab):
        """

        :param vocab: 已有的词汇表（语料库）
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        处理输入文本，将其转换为词元ID
        :param text:
        :return:
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # 处理未知单词，用<|unk|>词元代替未知单词
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        将词元ID转换回文本
        :param ids:
        :return:
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # 移除特定标点符号前的空格
        # text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt)
        # 使用滑动窗口将文本划分为长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self, ):
        """
        返回数据集的总行数
        :return:
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        返回数据集的指定行
        :param idx:
        :return:
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    用于批量生成 输入-目标对的数据加载器
    :param txt:
    :param batch_size:
    :param max_length:
    :param stride:
    :param shuffle:
    :param drop_last:
    :param num_workers:
    :return:
    """

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def softmax_naive(x):
    """
    用于归一化注意力分数的 softmax 函数的基础实现
    这种实现方式，处理大输入值或小输入值时可能会遇到数值稳定性问题，比如溢出和下溢
    :param x:
    :return:
    """
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def pytorch_softmax(x):
    return torch.softmax(x, dim=0)


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算交叉熵损失
    :param input_batch:
    :param target_batch:
    :param model:
    :param device:
    :return:
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算训练集和验证集损失得函数
    :param data_loader:
    :param model:
    :param device:
    :param num_batches:
    :return:
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 每批次的损失的总和
            total_loss += loss.item()
        else:
            break
    # 对所有批次的损失求平均值
    return total_loss / num_batches







if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    # preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    # preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # # 去除词汇表重复词元
    # all_words = sorted(set(preprocessed))
    # # 添加特殊词元
    # all_words.extend(["<|endoftext|>", "<|unk|>"])
    # # 创建词汇表
    # vocab = {token: integer for integer, token in enumerate(all_words)}
    # # print(len(vocab.items()))
    #
    # tokenizer = SimpleTokenizerV1(vocab)
    # text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    #
    # ids = tokenizer.encode(text)
    # print(ids)
    # # 解码，利用词汇表将词元id转为对应的文本信息
    # print(tokenizer.decode(ids))

    # dataloader = create_dataloader_v1(raw_text,batch_size=1,max_length=4,
    #                                   stride=1,shuffle=False)
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # second_batch = next(data_iter)
    # print(first_batch)
    # print(second_batch)

    # 以大于1的批次大小使用数据加载其进行采样
    # dataloader = create_dataloader_v1(raw_text,batch_size=8,max_length=4,
    #                                   stride=4,shuffle=False)
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # second_batch = next(data_iter)
    # print(first_batch)
    # print(second_batch)

    # torch.manual_seed(123)
    # # embedding_layer = torch.nn.Embedding(6,3)
    # embedding_layer = torch.nn.Embedding(50257,256)
    # print(embedding_layer.weight)

    # 实例化数据加载器
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print(f"Token IDs: {inputs}\n")
    # print(f"Target IDs: {targets}\n")

    # 使用嵌入层将这些词元ID嵌入256维的向量中
    vocab_size = 502257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print(f"Token embedding shape: {token_embeddings.shape}")

    # 获取GPT模型所采用的绝对位置嵌入，只需创建一个维度与token_embedding_layer相同的嵌入层即可
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(f"Pos embedding shape: {pos_embeddings.shape}")
    # 接下来，将这些向量直接添加到词元嵌入中。
    # PyTorch 会在每个批次中的每个 4×256 维的词元嵌入张量上都添加一个 4×256 维的pos_embeddings 张量
    input_embeddings = token_embeddings + pos_embeddings
    print(f"Input embedding shape: {input_embeddings.shape}")
