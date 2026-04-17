import torch

d_in = 5
d_out = 5

torch.manual_seed(123)
# 查询权重矩阵
w_q = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)
# 键权重矩阵
w_k = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)
# 值权重矩阵
w_v = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)
# 计算查询向量、键向量、值向量




