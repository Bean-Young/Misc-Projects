import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from node2vec import Node2Vec
import seaborn as sns
from collections import defaultdict

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 创建DataFrame以便处理
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = [target_names[i] for i in y]

# 2. 构建鸢尾花知识图谱
def build_iris_knowledge_graph(df):
    G = nx.Graph()
    
    # 添加类别节点
    species_nodes = {}
    for species in target_names:
        species_node = f"Species_{species}"
        G.add_node(species_node, type='species', label=species)
        species_nodes[species] = species_node
    
    # 添加特征节点
    feature_nodes = {}
    for feature in feature_names:
        # 添加特征类别节点
        feature_node = f"Feature_{feature.split(' ')[0]}"
        G.add_node(feature_node, type='feature', label=feature)
        feature_nodes[feature] = feature_node
        
        # 添加特征值节点（低、中、高）
        for level in ['Low', 'Medium', 'High']:
            level_node = f"{feature_node}_{level}"
            G.add_node(level_node, type='feature_value', feature=feature, level=level)
            G.add_edge(feature_node, level_node, relation='has_value')
    
    # 添加类别与特征的关联
    # 基于实际数据统计添加关系
    species_features = defaultdict(dict)
    for species in target_names:
        species_data = df[df['species_name'] == species]
        for feature in feature_names:
            # 计算该特征的统计信息
            mean_val = species_data[feature].mean()
            std_val = species_data[feature].std()
            
            # 添加关系边
            species_node = species_nodes[species]
            feature_node = feature_nodes[feature]
            
            # 添加统计关系
            G.add_edge(species_node, feature_node, 
                       relation='has_feature', 
                       mean=mean_val, 
                       std=std_val)
            
            # 存储特征统计信息
            species_features[species][feature] = {'mean': mean_val, 'std': std_val}
    
    return G, species_features

# 构建知识图谱
kg, species_features = build_iris_knowledge_graph(df)

# 3. 可视化知识图谱
def visualize_knowledge_graph(G):
    plt.figure(figsize=(15, 10))
    
    # 根据节点类型设置颜色
    node_colors = []
    for node in G.nodes:
        if G.nodes[node]['type'] == 'species':
            node_colors.append('lightgreen')
        elif G.nodes[node]['type'] == 'feature':
            node_colors.append('lightblue')
        else:
            node_colors.append('lightcoral')
    
    # 绘制图谱
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5)
    
    # 添加标签
    labels = {node: G.nodes[node].get('label', node.split('_')[-1]) for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title("Iris Dataset Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("iris_knowledge_graph.png", dpi=300)
    plt.close()

# 可视化图谱
visualize_knowledge_graph(kg)

# 4. 使用知识图谱增强特征
def enhance_features_with_kg(X, y, kg, species_features, target_names):
    # 创建特征矩阵
    enhanced_X = np.zeros((X.shape[0], X.shape[1] + len(target_names)))
    
    # 保留原始特征
    enhanced_X[:, :X.shape[1]] = X
    
    # 添加基于知识图谱的特征
    for i in range(X.shape[0]):
        features = X[i]
        species_idx = y[i]
        species = target_names[species_idx]
        
        # 计算特征值与类别典型值的相似度
        for j, feature_name in enumerate(feature_names):
            feature_val = features[j]
            species_mean = species_features[species][feature_name]['mean']
            species_std = species_features[species][feature_name]['std']
            
            # 计算标准化距离（考虑标准差）
            distance = abs(feature_val - species_mean) / (species_std + 1e-8)
            similarity = 1 / (1 + distance)
            
            # 添加到增强特征矩阵
            enhanced_X[i, X.shape[1] + species_idx] += similarity
    
    return enhanced_X

# 增强特征
enhanced_X = enhance_features_with_kg(X, y, kg, species_features, target_names)

# 5. 使用图嵌入技术
def generate_graph_embeddings(G, dimensions=8):
    # 生成图嵌入
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # 创建嵌入矩阵
    embeddings = {}
    for node in G.nodes:
        embeddings[node] = model.wv[node]
    
    return embeddings

# 生成图嵌入
embeddings = generate_graph_embeddings(kg)

# 6. 基于图嵌入增强特征
def enhance_features_with_embeddings(X, y, embeddings, target_names):
    # 获取类别节点的嵌入
    species_embeddings = {}
    for species in target_names:
        node = f"Species_{species}"
        species_embeddings[species] = embeddings[node]
    
    # 创建增强特征矩阵
    embedding_dim = len(next(iter(embeddings.values())))
    enhanced_X = np.zeros((X.shape[0], X.shape[1] + embedding_dim))
    
    # 保留原始特征
    enhanced_X[:, :X.shape[1]] = X
    
    # 添加嵌入特征
    for i in range(X.shape[0]):
        species_idx = y[i]
        species = target_names[species_idx]
        enhanced_X[i, X.shape[1]:] = species_embeddings[species]
    
    return enhanced_X

# 使用图嵌入增强特征
embedding_enhanced_X = enhance_features_with_embeddings(X, y, embeddings, target_names)

# 7. 数据预处理
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 使用增强特征
X_train_enhanced, X_test_enhanced, _, _ = train_test_split(
    enhanced_X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_embed, X_test_embed, _, _ = train_test_split(
    embedding_enhanced_X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler_enhanced = StandardScaler()
X_train_enhanced = scaler_enhanced.fit_transform(X_train_enhanced)
X_test_enhanced = scaler_enhanced.transform(X_test_enhanced)

scaler_embed = StandardScaler()
X_train_embed = scaler_embed.fit_transform(X_train_embed)
X_test_embed = scaler_embed.transform(X_test_embed)

# 8. 训练和评估模型
def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    return model, accuracy

# 使用不同特征集和模型
models = {
    "Basic RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KG-Enhanced RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Embedding-Enhanced RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Basic SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KG-Enhanced SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Embedding-Enhanced SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

# 训练和评估所有模型
results = {}
for name, model in models.items():
    if "Basic" in name:
        _, acc = train_and_evaluate(X_train, X_test, y_train, y_test, model, name)
        results[name] = acc
    elif "KG-Enhanced" in name:
        _, acc = train_and_evaluate(X_train_enhanced, X_test_enhanced, y_train, y_test, model, name)
        results[name] = acc
    else:  # Embedding-Enhanced
        _, acc = train_and_evaluate(X_train_embed, X_test_embed, y_train, y_test, model, name)
        results[name] = acc

# 9. 可视化特征空间
def visualize_feature_space(X, y, title, filename):
    # 使用PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建DataFrame
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Species'] = [target_names[i] for i in y]
    
    # 可视化
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_df, 
                    palette='viridis', s=100, alpha=0.8)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# 可视化不同特征空间
visualize_feature_space(X, y, "Original Feature Space", "original_feature_space.png")
visualize_feature_space(enhanced_X, y, "KG-Enhanced Feature Space", "kg_enhanced_feature_space.png")
visualize_feature_space(embedding_enhanced_X, y, "Embedding-Enhanced Feature Space", "embedding_enhanced_feature_space.png")

# 10. 可视化模型性能比较
def visualize_model_comparison(results):
    # 准备数据
    model_names = list(results.keys())
    accuracies = list(results.values())
    
    # 创建DataFrame
    df = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
    
    # 可视化
    plt.figure(figsize=(12, 6))
    bars = plt.barh(model_names, accuracies, color=['skyblue' if 'Basic' in n else 
                                                  'lightgreen' if 'KG-Enhanced' in n else 
                                                  'salmon' for n in model_names])
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', ha='left', va='center')
    
    plt.xlabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png", dpi=300)
    plt.close()

# 可视化模型比较
visualize_model_comparison(results)
