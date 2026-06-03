"""
在无标签数据上进行预训练
"""

import torch
import tiktoken

from src.pytorch_stu.build_llm_from_scratch.gpt_from_scratch import GPTModel, generate_text_simple
from src.pytorch_stu.build_llm_from_scratch.utils import create_dataloader_v1, calc_loss_loader, calc_loss_batch

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    # 将上下文长度从 1024 个词元缩短到 256 个词元
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    # 可以将 dropout 设置为 0，这也比较常见
    "drop_rate": 0.1,
    "qkv_bias": False
}


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 使用.unsqueeze(0)添加 batch 维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # 移除 batch 维度
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.eval()
#
# start_context = "Every effort moves you"
# tokenizer = tiktoken.get_encoding("gpt2")
#
# token_ids = generate_text_simple(
#      model=model,
#      idx=text_to_token_ids(start_context, tokenizer),
#      max_new_tokens=10,
#      context_size=GPT_CONFIG_124M["context_length"]
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

""" 计算训练集和验证集的损失 """

tokenizer = tiktoken.get_encoding("gpt2")
file_path = "./the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_token = len(tokenizer.encode(text_data))
# print(f"total characters: {total_characters}")
# print(f"total token: {total_token}")

# 将数据集分为训练集（90%）和验证集（10%）
# 训练数据占比
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
# 切出训练集
train_data = text_data[:split_idx]
# 切出验证集
val_data = text_data[split_idx:]

# 数据加载器
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# # model.eval()
# model.to(device)
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     # 使用损失梯度
#     val_loss = calc_loss_loader(val_loader, model, device)
# print(f"Training loss: {train_loss}")
# print(f"Validation loss: {val_loss}")

""" 训练大语言模型 """


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 模型评估模式，在评估阶段禁用 dropout，以产出稳定且可复现的结果
    model.eval()
    # 评估阶段也会禁用梯度跟踪，因为这是不需要的，而且这样可以减少计算开销
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    #     将模型重置为训练模式
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    # 禁用dropout
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50,
                                         context_size=context_size,temperature=1.4,top_k=25)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       num_epochs,
                       eval_freq,
                       eval_iter,
                       start_context,
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的损失梯度
            optimizer.zero_grad()
            # 计算每批次的损失梯度，交叉熵损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 反向传播，计算张量梯度
            loss.backward()
            # 执行单步优化，并使用计算的梯度（反向传播）更新权重
            optimizer.step()
            # 获取参数数量
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                # 每次模型权重更新后打印训练集和验证集的损失，以便可以评估训练是否改善了模型性能。
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
# 训练大模型
train_losses, val_losses, token_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)




