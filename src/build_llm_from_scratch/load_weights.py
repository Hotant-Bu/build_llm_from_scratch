"""
加载OpenAIGPT-2的权重
"""
import numpy as np
import tiktoken
import torch.nn

# import urllib.request
# url = (
#  "https://raw.githubusercontent.com/rasbt/"
#  "LLMs-from-scratch/main/ch05/"
#  "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)

from gpt_download import download_and_load_gpt2
from src.pytorch_stu.build_llm_from_scratch.gpt_from_scratch import GPTModel, generate_text_simple
from src.pytorch_stu.build_llm_from_scratch.train_without_label import GPT_CONFIG_124M, text_to_token_ids, \
    token_ids_to_text

settings, params = download_and_load_gpt2(model_size="124M",models_dir="gpt2")

print(f"settings: {settings}")
print(f"params keys: {params.keys()}")
# print(f"params embedding weights: {params['wte']}")

def assign(left, right):
    """
    检查两个张量或数组（left 和 right）是否具有相同的维度或形
    状，并将 right 张量返回为可训练的 PyTorch 参数
    :param left:
    :param right:
    :return:
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape} Right: {right.shape}")

    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt:GPTModel,params):
    """
    将模型的位置信息和词元嵌入权重
    设置为 params 中指定的值
    :param gpt:
    :param params:
    :return:
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params['wte'])

    for b in range(len(params['blocks'])):
        # 获取q、k、v的初始化权重矩阵
        q_w, k_w,v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"],3,axis=-1)

        # 设置transformer注意力块权重（Q、K、V）
        gpt.trf_blocks[b].att.w_query.weight = assign(gpt.trf_blocks[b].att.w_query.weight, q_w.T)
        gpt.trf_blocks[b].att.w_key.weight = assign(gpt.trf_blocks[b].att.w_key.weight, k_w.T)
        gpt.trf_blocks[b].att.w_value.weight = assign(gpt.trf_blocks[b].att.w_value.weight, v_w.T)

        # 设置偏置
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_query.bias = assign(gpt.trf_blocks[b].att.w_query.bias, q_b)
        gpt.trf_blocks[b].att.w_key.bias = assign(gpt.trf_blocks[b].att.w_key.bias, k_b)
        gpt.trf_blocks[b].att.w_value.bias = assign(gpt.trf_blocks[b].att.w_value.bias, v_b)

        # 将参数中的注意力线性层（投影）偏置设置赋值给gpt中对应的矩阵设置
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,
                                                       params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,
                                                     params["blocks"][b]["attn"]["c_proj"]["b"])

        # 将参数中的前馈神经网络输出要用到的线性层（投影）偏置设置赋值给gpt中对应的矩阵设置（包含权重和偏置）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,
                                                       params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,
                                                     params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,
                                                       params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,
                                                     params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 设置归一化处理的参数
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale,
                                               params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift,
                                               params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale,
                                               params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift,
                                               params["blocks"][b]["ln_2"]["b"])
        # 设置最终层归一化参数
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        # 设置输出的线性层参数。经过线性层投影（本质上是加权求和），汇总多头注意力
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

model_configs = {
 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
# gpt.eval()
# 加载gpt的参数
load_weights_into_gpt(gpt,params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
token_ids = generate_text_simple(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Load GPT weights Output text:\n", token_ids_to_text(token_ids, tokenizer))















