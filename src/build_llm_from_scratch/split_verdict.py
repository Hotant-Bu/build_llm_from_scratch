import re

with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

# print("Total number of character: ", len(raw_text))

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

preprocessed = [ item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))

# print(preprocessed[:30])

# 去除词汇表重复词元
all_words = sorted(set(preprocessed))
vacab_size = len(all_words)
print(vacab_size)

# 创建词汇表
vocab = { token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break






