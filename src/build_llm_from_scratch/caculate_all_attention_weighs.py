import torch

from src.pytorch_stu.build_llm_from_scratch.utils import pytorch_softmax

# 输入词元的向量表示
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts (x^3)
     [0.22, 0.58, 0.33],  # with (x^4)
     [0.77, 0.25, 0.10],  # one (x^5)
     [0.05, 0.80, 0.55]]  # step (x^6)
)

# # 计算所有输入词元的注意力权重
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)
# # 归一化
# attn_scores_softmax = pytorch_softmax(attn_scores)
# print(attn_scores_softmax)
# print(attn_scores_softmax.sum())


# 使用矩阵乘法计算。@符号代表矩阵乘法，
# inputs @ inputs.T 计算行向量之间的相似度
# inputs.T @ inputs 计算列之间的相似度
attn_scores_1 = inputs @ inputs.T
print(attn_scores_1)
print(attn_scores_1.sum())
# 归一化
attn_scores_1_normalize = pytorch_softmax(attn_scores_1)
print(attn_scores_1_normalize.sum())











