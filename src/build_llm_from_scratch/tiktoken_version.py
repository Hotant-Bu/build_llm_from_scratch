import tiktoken
print(f"tiktoken version: {tiktoken.__version__}")

# 实例化tiktoken中的BPE分词器
tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#  "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#  "of someunknownPlace."
# )
#
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# print(tokenizer.decode(integers))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

encode_text = tokenizer.encode(raw_text)
print(len(encode_text))









