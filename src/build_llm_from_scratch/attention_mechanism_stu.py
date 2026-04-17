import torch

from src.pytorch_stu.build_llm_from_scratch.utils import softmax_naive, pytorch_softmax

# 输入词元的向量表示
inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)
query = inputs[1]
print(f"query: {query}")
attn_scores_2 = torch.empty(inputs.shape[0])
print(f"attn_scores_2: {attn_scores_2}")
for i, x_i in enumerate(inputs):
    print(f"x_i: {x_i}")
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# 归一化处理。归一化的主要目的是获得总和为 1 的注意力权重。
attn_weights_2_tmp= attn_scores_2/attn_scores_2.sum()
print(f"Attention weights: {attn_weights_2_tmp}")
print(f"Sum: {attn_weights_2_tmp.sum()}")

# 使用softmax函数进行归一化
# attn_weights_2_naive = softmax_naive(attn_scores_2)
attn_weights_2_naive = pytorch_softmax(attn_scores_2)
print(f"Attention weights: {attn_weights_2_naive}")
print(f"Attention weights Sum: {attn_weights_2_naive.sum()}")


