import torch

from src.pytorch_stu.build_llm_from_scratch.utils import create_dataloader_v1


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 实例化数据加载器
max_length = 4
dataloader = create_dataloader_v1(
    raw_text ,batch_size=8 ,max_length=max_length ,stride=max_length ,shuffle=False
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
pos_embedding_layer = torch.nn.Embedding(context_length ,output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"Pos embedding shape: {pos_embeddings.shape}")
# 接下来，将这些向量直接添加到词元嵌入中。
# PyTorch 会在每个批次中的每个 4×256 维的词元嵌入张量上都添加一个 4×256 维的pos_embeddings 张量
input_embeddings = token_embeddings + pos_embeddings
print(f"Input embedding shape: {input_embeddings.shape}")



