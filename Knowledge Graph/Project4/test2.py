import pandas as pd
import numpy as np
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

# 设置全局中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免负号乱码
import seaborn as sns
import nltk
nltk.data.path.append('/home/yyz/KG-Class/Project4/nltk_data')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# 确保结果目录存在
result_dir = "/home/yyz/KG-Class/Project4/Result"
os.makedirs(result_dir, exist_ok=True)


# 模拟数据集（扩展数据集以提高模型性能）
data = {
    "news": [
        "The company reported a 25% increase in earnings this quarter.",
        "There is a scandal involving the CFO that might impact the stock negatively.",
        "New product launch has been a massive success, leading to a sharp increase in sales.",
        "The Federal Reserve announced interest rate hikes, causing market uncertainty.",
        "Analysts upgrade the stock to 'buy' rating after strong financial results.",
        "Competitor releases superior product, threatening market share.",
        "Company announces record-breaking profits and dividend increase.",
        "Supply chain disruptions expected to impact next quarter's earnings.",
        "Major partnership signed with industry leader, opening new markets.",
        "Regulatory investigation launched into company's business practices.",
        "Positive consumer response to new advertising campaign boosts brand image.",
        "Unexpected CEO resignation shakes investor confidence.",
        "Global expansion plans accelerated due to strong demand.",
        "Product recall announced due to safety concerns.",
        "Stock split announced to make shares more accessible to retail investors.",
        "Industry-wide price war negatively impacts profit margins.",
        "Successful patent application secures competitive advantage.",
        "Data breach incident compromises customer information.",
        "Favorable court ruling removes legal overhang.",
        "Economic downturn forecasts lead to sector-wide sell-off."
    ],
    "open_price": [100.25, 150.30, 120.50, 130.75, 125.60, 110.45, 135.80, 128.90, 140.20, 132.15,
                   118.75, 142.60, 125.30, 115.80, 138.40, 122.50, 145.20, 119.30, 131.80, 120.45],
    "close_price": [110.30, 140.20, 130.75, 125.40, 135.90, 105.80, 145.60, 123.50, 155.40, 128.75,
                    125.40, 132.80, 138.90, 108.60, 148.20, 115.30, 158.40, 112.80, 142.50, 112.30],
    "volume": [2000, 3000, 2500, 3500, 2800, 3200, 2400, 2900, 2700, 3300,
               2600, 3800, 3100, 3600, 2900, 3400, 2750, 4200, 3000, 3700],
    "next_day_price_change": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
                              1, -1, 1, -1, 1, -1, 1, -1, 1, -1]  # 1 for increase, -1 for decrease
}

# 创建DataFrame
df = pd.DataFrame(data)

# 添加技术指标特征
def add_technical_indicators(df):
    # 计算价格变化百分比
    df['price_change'] = ((df['close_price'] - df['open_price']) / df['open_price']) * 100
    
    # 计算5日简单移动平均
    df['sma_5'] = df['close_price'].rolling(window=5).mean()
    
    # 计算相对强弱指数 (RSI)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算波动率
    df['volatility'] = df['close_price'].rolling(window=5).std()
    
    # 填充NaN值
    df.fillna(0, inplace=True)
    return df

# 文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 分词
    words = nltk.word_tokenize(text)
    
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# 情感分析函数
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # 情感极性: -1(负面) 到 1(正面)
    polarity = analysis.sentiment.polarity
    # 主观性: 0(客观) 到 1(主观)
    subjectivity = analysis.sentiment.subjectivity
    
    # 创建情感标签
    if polarity > 0.1:
        sentiment = 1  # 正面
    elif polarity < -0.1:
        sentiment = -1  # 负面
    else:
        sentiment = 0  # 中性
    
    return polarity, subjectivity, sentiment

# 添加技术指标
df = add_technical_indicators(df)

# 文本预处理
df['cleaned_news'] = df['news'].apply(preprocess_text)

# 应用情感分析
sentiment_results = df['cleaned_news'].apply(analyze_sentiment)
df[['polarity', 'subjectivity', 'sentiment']] = pd.DataFrame(sentiment_results.tolist(), index=df.index)

# 添加文本特征 - TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=25, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_news'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.columns = ['tfidf_' + col for col in tfidf_df.columns]

# 添加文本特征 - 词袋模型
bow_vectorizer = CountVectorizer(max_features=20, stop_words='english')
bow_features = bow_vectorizer.fit_transform(df['cleaned_news'])
bow_df = pd.DataFrame(bow_features.toarray(), columns=bow_vectorizer.get_feature_names_out())
bow_df.columns = ['bow_' + col for col in bow_df.columns]

# 合并所有特征
feature_df = pd.concat([
    df[['open_price', 'close_price', 'volume', 'price_change', 'sma_5', 'rsi', 'volatility',
        'polarity', 'subjectivity', 'sentiment']],
    tfidf_df,
    bow_df
], axis=1)

# 目标变量
target = df['next_day_price_change']

# 特征缩放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df)
scaled_feature_df = pd.DataFrame(scaled_features, columns=feature_df.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    scaled_feature_df, target, test_size=0.2, random_state=42
)

# 训练随机森林模型
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    max_depth=10,
    min_samples_split=5
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("模型评估报告:")
print(report)
print(f"准确率: {accuracy:.2f}")

# 特征重要性分析
feature_importances = pd.DataFrame({
    'Feature': feature_df.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(feature_importances.head(10))

# 保存结果到文件
def save_results():
    # 保存模型评估报告
    with open(os.path.join(result_dir, 'model_report.txt'), 'w') as f:
        f.write("股票价格变动预测模型评估报告\n")
        f.write("===================================\n\n")
        f.write(report)
        f.write(f"\n准确率: {accuracy:.4f}")
    
    # 保存特征重要性
    feature_importances.to_csv(os.path.join(result_dir, 'feature_importances.csv'), index=False)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '新闻': df.loc[X_test.index, 'news'],
        '实际价格变动': y_test,
        '预测价格变动': y_pred,
        '正确': y_test == y_pred
    })
    results_df.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)
    
    # 创建可视化并保存到文件
    plt.figure(figsize=(12, 8))
    
    # 1. 情感极性分布
    plt.subplot(2, 2, 1)
    sns.histplot(df['polarity'], bins=20, kde=True)
    plt.title('新闻情感极性分布')
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('情感极性 (-1: 负面, 1: 正面)')
    
    # 2. 价格变动与情感关系
    plt.subplot(2, 2, 2)
    sns.boxplot(x='next_day_price_change', y='polarity', data=df)
    plt.title('价格变动与情感极性关系')
    plt.xticks([-1, 1], ['下跌', '上涨'])
    plt.xlabel('价格变动')
    
    # 3. 特征重要性
    plt.subplot(2, 2, 3)
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
    plt.title('Top 10 重要特征')
    plt.xlabel('重要性')
    
    # 4. 混淆矩阵
    plt.subplot(2, 2, 4)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['下跌', '上涨'], yticklabels=['下跌', '上涨'])
    plt.title('混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('实际')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'analysis_plots.png'))
    plt.close()
    
    # 保存实际vs预测结果图
    plt.figure(figsize=(10, 6))
    result_plot = pd.DataFrame({'实际': y_test, '预测': y_pred})
    result_plot = result_plot.reset_index(drop=True)
    result_plot.plot(marker='o', linestyle='-')
    plt.title('实际 vs 预测价格变动')
    plt.xlabel('样本索引')
    plt.ylabel('价格变动 (1:上涨, -1:下跌)')
    plt.legend()
    plt.axhline(0, color='gray', linestyle='--')
    plt.savefig(os.path.join(result_dir, 'actual_vs_predicted.png'))
    plt.close()

# 保存所有结果
save_results()
