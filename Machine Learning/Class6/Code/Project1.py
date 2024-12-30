import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from BPNet import BPNetwork

# 读取数据时指定没有表头
positive_data = pd.read_csv("data/positive.csv", header=None)
negative_data = pd.read_csv("data/negative.csv", header=None)
positive_data = positive_data.T
negative_data = negative_data.T

# 添加统一的列名
num_columns = positive_data.shape[1]
column_names = [f"Feature_{i}" for i in range(num_columns)]
positive_data.columns = column_names
negative_data.columns = column_names

# 拼接数据
data = pd.concat([positive_data, negative_data], axis=0).reset_index(drop=True)
labels = np.concatenate([np.ones(len(positive_data)), np.zeros(len(negative_data))])

# (一) UMAP 可视化 (原始数据)
reducer = umap.UMAP(n_components=2, random_state=42)
reduced_data = reducer.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:200, 0], reduced_data[:200, 1], label="Positive", alpha=0.7)
plt.scatter(reduced_data[200:, 0], reduced_data[200:, 1], label="Negative", alpha=0.7)
plt.title("UMAP Visualization of Original Data")
plt.legend()
plt.tight_layout()
plt.savefig("figure/UMAP_OriginalData.png", dpi=300)
plt.show()


# (二) 固定五折交叉验证（手动切分）
positive_indices = np.arange(0, 200)
negative_indices = np.arange(200, 400)

# 将正负样本各分成 5 份，每份 40 个样本
pos_splits = [positive_indices[i*40:(i+1)*40] for i in range(5)]
neg_splits = [negative_indices[i*40:(i+1)*40] for i in range(5)]
subsets = []
for i in range(5):
    test_indices = np.concatenate([pos_splits[i], neg_splits[i]])
    train_indices = np.concatenate(
        [pos_splits[j] for j in range(5) if j != i] +
        [neg_splits[j] for j in range(5) if j != i]
    )
    subsets.append((test_indices, train_indices))

all_auroc_fixed = []

# 训练并测试 (固定五折)
for fold, (test_indices, train_indices) in enumerate(subsets):
    train_data, test_data = data.iloc[train_indices], data.iloc[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    # 转换为 Tensor
    train_data = torch.tensor(train_data.values, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    test_data = torch.tensor(test_data.values, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    # 模型实例化
    model = BPNetwork(input_dim=train_data.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    with torch.no_grad():
        predictions = model(test_data).numpy()
        auroc = roc_auc_score(test_labels.numpy(), predictions)
        all_auroc_fixed.append(auroc)

        # 倒数第二层的输出 (2维)
        intermediate_layer = nn.Sequential(*list(model.model.children())[:-2])
        reduced_test_data = intermediate_layer(test_data).numpy()

        # 绘制并保存测试集散点图
        plt.figure(figsize=(12, 5))

        # 子图1：预测标签
        plt.subplot(1, 2, 1)
        plt.scatter(reduced_test_data[:, 0],
                    reduced_test_data[:, 1],
                    c=(predictions > 0.5).squeeze(),
                    cmap="coolwarm")
        plt.title(f"Fold {fold+1}: Predicted Labels")

        # 子图2：真实标签
        plt.subplot(1, 2, 2)
        plt.scatter(reduced_test_data[:, 0],
                    reduced_test_data[:, 1],
                    c=test_labels.numpy().squeeze(),
                    cmap="coolwarm")
        plt.title(f"Fold {fold+1}: True Labels")

        plt.tight_layout()
        save_path = f"figure/Fold_{fold+1}_TestScatter.png"
        plt.savefig(save_path, dpi=300)
        plt.show()

fixed_mean_auroc = np.mean(all_auroc_fixed)
print(f"[固定五折] Average AUROC across folds: {fixed_mean_auroc:.4f}")


# (三) 特征选择（方差最大）
variances = data.var(axis=0)
top_features = variances.nlargest(2).index
selected_data = data[top_features]

plt.figure(figsize=(8, 6))
plt.scatter(selected_data.iloc[:200, 0], selected_data.iloc[:200, 1],
            label="Positive", alpha=0.7)
plt.scatter(selected_data.iloc[200:, 0], selected_data.iloc[200:, 1],
            label="Negative", alpha=0.7)
plt.title("Scatter Plot after Feature Selection")
plt.legend()
plt.tight_layout()
plt.savefig("figure/FeatureSelection_2D.png", dpi=300)
plt.show()

# (四) PCA降维到二维
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:200, 0], pca_data[:200, 1], label="Positive", alpha=0.7)
plt.scatter(pca_data[200:, 0], pca_data[200:, 1], label="Negative", alpha=0.7)
plt.title("Scatter Plot after PCA")
plt.legend()
plt.tight_layout()
plt.savefig("figure/PCA_2D.png", dpi=300)
plt.show()


# (五) 多次随机划分的五折交叉验证（与固定划分作对比）
num_runs = 5  # 做 5 次随机划分
all_auroc_runs = []  # 存储每次随机五折的平均 AUROC

# 将 DataFrame 转成 numpy 方便 KFold 索引
data_np = data.values
labels_np = labels

for run_id in range(num_runs):
    # 这里通过 KFold shuffle=True 并设置不同的 random_state
    kf = KFold(n_splits=5, shuffle=True, random_state=run_id)
    
    run_auroc_list = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(data_np)):
        # 拆分数据
        X_train = torch.tensor(data_np[train_idx], dtype=torch.float32)
        X_test  = torch.tensor(data_np[test_idx],  dtype=torch.float32)
        y_train = torch.tensor(labels_np[train_idx], dtype=torch.float32).unsqueeze(1)
        y_test  = torch.tensor(labels_np[test_idx],  dtype=torch.float32).unsqueeze(1)
        
        # 创建 BPNetwork
        model = BPNetwork(input_dim=X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).numpy()
            auroc = roc_auc_score(y_test.numpy(), predictions)
            run_auroc_list.append(auroc)
    
    mean_auroc = np.mean(run_auroc_list)
    all_auroc_runs.append(mean_auroc)
    print(f"[随机划分第 {run_id+1} 次] 5-Fold Average AUROC: {mean_auroc:.4f}")

print("随机多次的 AUROC 结果:", all_auroc_runs)
print("随机多次平均 AUROC:", np.mean(all_auroc_runs))