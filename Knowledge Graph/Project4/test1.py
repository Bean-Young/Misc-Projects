import re
import nltk
nltk.data.path.append('/home/yyz/KG-Class/Project4/nltk_data')
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import matplotlib.pyplot as plt
import matplotlib
# 设置全局中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免负号乱码
import seaborn as sns
import numpy as np
import os
import itertools

# 确保结果目录存在
result_dir = "/home/yyz/KG-Class/Project4/Result"
os.makedirs(result_dir, exist_ok=True)


# 原始数据（扩展数据集以提高模型性能）
data = [
    ("John Smith is the CEO of Apple Inc.", {"entities": [(0, 10, "PERSON"), (28, 35, "ORG")]}),
    ("I live in New York City", {"entities": [(10, 22, "GPE")]}),
    ("Microsoft Corporation is headquartered in Redmond, Washington", {"entities": [(0, 21, "ORG"), (44, 52, "GPE")]}),
    ("Elon Musk is the founder of SpaceX and Tesla, Inc.", {"entities": [(0, 9, "PERSON"), (35, 41, "ORG"), (46, 51, "ORG")]}),
    ("Paris is the capital of France.", {"entities": [(0, 5, "GPE"), (24, 30, "GPE")]}),
    # 新增数据
    ("Amazon.com, Inc. is located in Seattle.", {"entities": [(0, 12, "ORG"), (30, 37, "GPE")]}),
    ("Tim Cook announced new products at Apple Park.", {"entities": [(0, 8, "PERSON"), (36, 46, "ORG")]}),
    ("Beijing and Shanghai are major cities in China.", {"entities": [(0, 7, "GPE"), (12, 20, "GPE"), (43, 48, "GPE")]}),
    ("Mark Zuckerberg leads Meta Platforms, Inc.", {"entities": [(0, 15, "PERSON"), (22, 39, "ORG")]}),
    ("London is the capital of the United Kingdom.", {"entities": [(0, 6, "GPE"), (30, 45, "GPE")]}),
    ("Google LLC announced new AI features at their headquarters in Mountain View.", {"entities": [(0, 10, "ORG"), (60, 72, "GPE")]}),
    ("President Joe Biden visited the White House today.", {"entities": [(10, 18, "PERSON"), (32, 43, "ORG")]}),
    ("Sundar Pichai is the CEO of Alphabet Inc.", {"entities": [(0, 12, "PERSON"), (30, 42, "ORG")]}),
    ("The Eiffel Tower is located in Paris, France.", {"entities": [(34, 39, "GPE"), (41, 48, "GPE")]}),
    ("Netflix Inc. produces original content in Los Gatos, California.", {"entities": [(0, 10, "ORG"), (45, 54, "GPE"), (56, 67, "GPE")]}),
]

# 辅助函数：将实体标注转换为单词级别的BIO标签
def convert_to_bio(sentence, entities):
    tokens = word_tokenize(sentence)
    # 获取每个token的字符偏移量
    start_positions = []
    end_positions = []
    current_pos = 0
    for token in tokens:
        start = sentence.find(token, current_pos)
        if start == -1:  # 处理找不到的情况
            start = current_pos
        end = start + len(token)
        start_positions.append(start)
        end_positions.append(end)
        current_pos = end
    
    # 初始化标签为O
    labels = ['O'] * len(tokens)
    
    # 标记实体
    for start_char, end_char, label_type in entities:
        entity_tokens = []
        for i, (token_start, token_end) in enumerate(zip(start_positions, end_positions)):
            # 检查token是否在实体范围内
            if (token_start >= start_char and token_end <= end_char) or \
               (token_start < start_char and token_end > start_char) or \
               (token_start < end_char and token_end > end_char):
                # 确定标签类型 (B- 或 I-)
                if not entity_tokens:  # 第一个token
                    prefix = 'B-'
                else:
                    # 检查是否与前一个token连续
                    if i == entity_tokens[-1] + 1:
                        prefix = 'I-'
                    else:
                        prefix = 'B-'
                labels[i] = prefix + label_type
                entity_tokens.append(i)
    return tokens, labels

# 特征提取函数（增强版）
def extract_features(sentence_tokens, index, pos_tags=None):
    token = sentence_tokens[index]
    
    features = {
        'bias': 1.0,
        'word.lower': token.lower(),
        'word[-3:]': token[-3:],
        'word[-2:]': token[-2:],
        'word[:3]': token[:3],
        'word[:2]': token[:2],
        'word.isupper': token.isupper(),
        'word.istitle': token.istitle(),
        'word.isdigit': token.isdigit(),
        'word.length': len(token),
        'word.contains_dash': '-' in token,
        'word.contains_dot': '.' in token,
    }
    
    # 添加上下文特征
    if index > 0:
        prev_token = sentence_tokens[index-1]
        features.update({
            'prev_word.lower': prev_token.lower(),
            'prev_word.istitle': prev_token.istitle(),
            'prev_word.isupper': prev_token.isupper(),
            'prev_word.isdigit': prev_token.isdigit(),
            'prev_word.length': len(prev_token),
        })
        if pos_tags:
            features['prev_pos'] = pos_tags[index-1][1]
    else:
        features['BOS'] = True  # 句子开始
        
    if index < len(sentence_tokens)-1:
        next_token = sentence_tokens[index+1]
        features.update({
            'next_word.lower': next_token.lower(),
            'next_word.istitle': next_token.istitle(),
            'next_word.isupper': next_token.isupper(),
            'next_word.isdigit': next_token.isdigit(),
            'next_word.length': len(next_token),
        })
        if pos_tags:
            features['next_pos'] = pos_tags[index+1][1]
    else:
        features['EOS'] = True  # 句子结束
        
    # 添加词性特征（如果提供）
    if pos_tags:
        features['pos'] = pos_tags[index][1]
        
    return features

# 将整个句子转换为特征序列
def sentence_to_features(sentence_tokens):
    # 获取词性标注
    pos_tags = pos_tag(sentence_tokens)
    return [extract_features(sentence_tokens, i, pos_tags) for i in range(len(sentence_tokens))]

# 可视化实体识别结果
def visualize_ner_results(sentence, true_entities, pred_entities, filename):
    plt.figure(figsize=(12, 4))
    
    # 创建文本显示
    text = sentence
    plt.text(0.5, 0.7, text, ha='center', va='center', fontsize=14, wrap=True)
    
    # 显示真实实体
    plt.text(0.5, 0.5, "True Entities:", ha='center', va='center', fontsize=12, weight='bold')
    entity_text = ", ".join([f"{sentence[s:e]} ({l})" for (s, e, l) in true_entities])
    plt.text(0.5, 0.4, entity_text, ha='center', va='center', fontsize=12)
    
    # 显示预测实体
    plt.text(0.5, 0.3, "Predicted Entities:", ha='center', va='center', fontsize=12, weight='bold')
    pred_text = ", ".join([f"{sentence[s:e]} ({l})" for (s, e, l) in pred_entities])
    plt.text(0.5, 0.2, pred_text, ha='center', va='center', fontsize=12, color='blue')
    
    # 移除坐标轴
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=300)
    plt.close()

# 主函数
def main():
    # 转换数据为CRF格式
    X = []  # 特征序列
    y = []  # 标签序列
    sentences = []  # 原始句子
    true_entities_list = []  # 真实实体列表
    
    for sentence, annotations in data:
        tokens, labels = convert_to_bio(sentence, annotations['entities'])
        X.append(sentence_to_features(tokens))
        y.append(labels)
        sentences.append(sentence)
        true_entities_list.append(annotations['entities'])
    
    # 划分训练集和测试集 (80%训练，20%测试)
    X_train, X_test, y_train, y_test, sent_train, sent_test, true_entities_train, true_entities_test = train_test_split(
        X, y, sentences, true_entities_list, test_size=0.2, random_state=42
    )
    
    # 训练CRF模型
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True,
        all_possible_states=True
    )
    crf.fit(X_train, y_train)
    
    # 在测试集上预测
    y_pred = crf.predict(X_test)
    
    # 评估性能
    labels = ['B-PERSON', 'I-PERSON', 'B-GPE', 'I-GPE', 'B-ORG', 'I-ORG']
    
    # 分类报告
    report = metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3
    )
    print("分类报告:")
    print(report)
    
    # 保存分类报告
    with open(os.path.join(result_dir, 'ner_classification_report.txt'), 'w') as f:
        f.write("命名实体识别分类报告\n")
        f.write("=======================\n\n")
        f.write(report)
    
    # 输出测试结果示例
    print("\n测试结果示例:")
    for i, (sentence, features, true_labels, pred_labels) in enumerate(zip(sent_test, X_test, y_test, y_pred)):
        tokens = [feat['word.lower'] for feat in features]
        print(f"\n句子 {i+1}: {sentence}")
        for token, true_label, pred_label in zip(tokens, true_labels, pred_labels):
            print(f"{token:<15} {true_label:<10} {pred_label:<10}")
        
        # 提取预测的实体
        pred_entities = []
        current_entity = None
        start_idx = 0
        char_pos = 0
        
        for j, token in enumerate(tokens):
            token_start = sentence.find(token, char_pos)
            if token_start == -1:
                token_start = char_pos
            token_end = token_start + len(token)
            char_pos = token_end
            
            if pred_labels[j].startswith('B-'):
                if current_entity:
                    pred_entities.append((start_idx, char_pos - len(token), current_entity.split('-')[1]))
                current_entity = pred_labels[j]
                start_idx = token_start
            elif pred_labels[j].startswith('I-'):
                if current_entity and current_entity.split('-')[1] == pred_labels[j].split('-')[1]:
                    continue
                else:
                    current_entity = None
            else:
                if current_entity:
                    pred_entities.append((start_idx, token_start, current_entity.split('-')[1]))
                    current_entity = None
        
        if current_entity:
            pred_entities.append((start_idx, char_pos, current_entity.split('-')[1]))
        
        # 可视化结果
        visualize_ner_results(sentence, true_entities_test[i], pred_entities, f'ner_result_{i+1}.png')
    
    # 修正的特征重要性分析
    print("\n特征重要性分析:")
    
    # 获取所有特征名称
    all_features = set()
    for sentence_features in X_train:
        for token_features in sentence_features:
            all_features.update(token_features.keys())
    
    # 排除特殊特征
    exclude_features = {'bias', 'BOS', 'EOS'}
    features_to_analyze = [f for f in all_features if f not in exclude_features]
    
    feature_importance = {}
    
    for feature in features_to_analyze:
        # 创建简化特征集
        simplified_X_train = []
        for sent_features in X_train:
            simplified_sent = []
            for token_features in sent_features:
                # 只保留当前特征和基本特征
                simplified_token = {'bias': 1.0}
                
                # 添加当前特征（如果存在）
                if feature in token_features:
                    simplified_token[feature] = token_features[feature]
                else:
                    # 如果特征不存在，使用默认值
                    if feature.startswith('prev_') or feature.startswith('next_'):
                        # 对于上下文特征，使用空字符串作为默认值
                        simplified_token[feature] = ''
                    else:
                        # 对于其他特征，使用特征特定的默认值
                        if feature == 'word.length':
                            simplified_token[feature] = 0
                        elif feature == 'word.isdigit' or feature == 'word.isupper' or feature == 'word.istitle':
                            simplified_token[feature] = False
                        else:
                            simplified_token[feature] = ''
                
                # 添加边界特征
                if 'BOS' in token_features:
                    simplified_token['BOS'] = True
                if 'EOS' in token_features:
                    simplified_token['EOS'] = True
                    
                simplified_sent.append(simplified_token)
            simplified_X_train.append(simplified_sent)
        
        # 训练简化模型
        crf_simple = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf_simple.fit(simplified_X_train, y_train)
        
        # 评估性能
        # 创建测试集的简化特征
        simplified_X_test = []
        for sent_features in X_test:
            simplified_sent = []
            for token_features in sent_features:
                simplified_token = {'bias': 1.0}
                
                if feature in token_features:
                    simplified_token[feature] = token_features[feature]
                else:
                    if feature.startswith('prev_') or feature.startswith('next_'):
                        simplified_token[feature] = ''
                    else:
                        if feature == 'word.length':
                            simplified_token[feature] = 0
                        elif feature == 'word.isdigit' or feature == 'word.isupper' or feature == 'word.istitle':
                            simplified_token[feature] = False
                        else:
                            simplified_token[feature] = ''
                
                if 'BOS' in token_features:
                    simplified_token['BOS'] = True
                if 'EOS' in token_features:
                    simplified_token['EOS'] = True
                    
                simplified_sent.append(simplified_token)
            simplified_X_test.append(simplified_sent)
        
        y_pred_simple = crf_simple.predict(simplified_X_test)
        accuracy = metrics.flat_accuracy_score(y_test, y_pred_simple)
        feature_importance[feature] = accuracy
    
    # 排序特征重要性
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n特征重要性排序:")
    for feature, importance in sorted_features:
        print(f"{feature:<25} {importance:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    features, importances = zip(*sorted_features)
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, importances, align='center', color='skyblue')
    plt.yticks(y_pos, features)
    plt.xlabel('准确率')
    plt.title('特征重要性分析')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'ner_feature_importance.png'), dpi=300)
    plt.close()
    
    # 标签分布可视化
    all_labels = list(itertools.chain.from_iterable(y))
    label_counts = {label: all_labels.count(label) for label in set(all_labels)}
    
    plt.figure(figsize=(10, 6))
    labels, counts = zip(*sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    plt.bar(labels, counts, color='lightgreen')
    plt.xlabel('标签')
    plt.ylabel('出现次数')
    plt.title('标签分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'ner_label_distribution.png'), dpi=300)
    plt.close()
    
    print(f"\n所有结果已保存到目录: {result_dir}")

if __name__ == "__main__":
    main()