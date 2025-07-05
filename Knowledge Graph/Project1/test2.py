# 题目 2：读取文本并统计每个句子的单词数
import nltk
nltk.data.path.append('/home/yyz/KG-Class/Project1/nltk_data') 
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

# 读取文本文件
with open('./KG-Class/Project1/test.txt', 'r') as file:
    text = file.read()

# 分句
sentences = sent_tokenize(text)

# 统计每句的单词数
print("每个句子的单词数（含符号）：")
for i, sentence in enumerate(sentences, 1):
    word_count = len(word_tokenize(sentence))
    print(f"句子 {i}: {word_count} 个单词")