import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# 设置随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)

# 创建结果目录
result_dir = "/home/yyz/KG-Class/Project6/Result"
os.makedirs(result_dir, exist_ok=True)

# 1. 数据收集和模拟
# 电影数据库
movies_db = {
    "The Avengers": {"genres": ["action", "adventure", "sci-fi"], "year": 2012, "rating": 8.0},
    "The Godfather": {"genres": ["crime", "drama"], "year": 1972, "rating": 9.2},
    "Titanic": {"genres": ["drama", "romance"], "year": 1997, "rating": 7.8},
    "The Shawshank Redemption": {"genres": ["drama"], "year": 1994, "rating": 9.3},
    "The Shining": {"genres": ["horror", "drama"], "year": 1980, "rating": 8.4},
    "Inception": {"genres": ["action", "adventure", "sci-fi"], "year": 2010, "rating": 8.8},
    "La La Land": {"genres": ["comedy", "drama", "romance"], "year": 2016, "rating": 8.0},
    "Get Out": {"genres": ["horror", "mystery"], "year": 2017, "rating": 7.7},
    "The Dark Knight": {"genres": ["action", "crime", "drama"], "year": 2008, "rating": 9.0},
    "Pulp Fiction": {"genres": ["crime", "drama"], "year": 1994, "rating": 8.9},
    "Forrest Gump": {"genres": ["drama", "romance"], "year": 1994, "rating": 8.8},
    "The Matrix": {"genres": ["action", "sci-fi"], "year": 1999, "rating": 8.7},
    "Parasite": {"genres": ["comedy", "drama", "thriller"], "year": 2019, "rating": 8.6},
    "Interstellar": {"genres": ["adventure", "drama", "sci-fi"], "year": 2014, "rating": 8.6},
    "Joker": {"genres": ["crime", "drama", "thriller"], "year": 2019, "rating": 8.4}
}

# 用户数据库
users_db = {
    "user1": {"action": 5, "comedy": 3, "drama": 4, "romance": 2, "horror": 1},
    "user2": {"action": 2, "comedy": 5, "drama": 3, "romance": 4, "horror": 1},
    "user3": {"action": 4, "comedy": 2, "drama": 5, "romance": 3, "horror": 4},
    "user4": {"action": 1, "comedy": 4, "drama": 3, "romance": 5, "horror": 2},
    "user5": {"action": 3, "comedy": 3, "drama": 4, "romance": 2, "horror": 5}
}

# 用户历史观影记录
user_history = {
    "user1": ["The Avengers", "The Dark Knight", "Inception"],
    "user2": ["La La Land", "Titanic", "Forrest Gump"],
    "user3": ["The Shawshank Redemption", "The Godfather", "Pulp Fiction"],
    "user4": ["Titanic", "La La Land", "Forrest Gump"],
    "user5": ["The Shining", "Get Out", "Joker"]
}

# 电影流行度数据（随时间变化）
movie_popularity = {
    movie: random.randint(1, 10) for movie in movies_db
}

# 2. 数据预处理
# 创建电影特征矩阵
movie_features = []
for movie, details in movies_db.items():
    # 创建特征向量 [动作, 喜剧, 剧情, 爱情, 恐怖, 年份(标准化), 评分(标准化)]
    features = [
        1 if "action" in details["genres"] else 0,
        1 if "comedy" in details["genres"] else 0,
        1 if "drama" in details["genres"] else 0,
        1 if "romance" in details["genres"] else 0,
        1 if "horror" in details["genres"] else 0,
        details["year"],
        details["rating"]
    ]
    movie_features.append(features)

movie_features = np.array(movie_features)
movie_names = list(movies_db.keys())

# 标准化年份和评分
scaler = MinMaxScaler()
movie_features[:, 5:] = scaler.fit_transform(movie_features[:, 5:])

# 3. 基于内容的推荐模型
def content_based_recommendation(user_prefs, movie_features, movie_names, top_n=5):
    """
    基于内容的推荐：根据用户偏好和电影特征的相似度进行推荐
    """
    # 用户偏好向量 [动作, 喜剧, 剧情, 爱情, 恐怖]
    user_vector = np.array([
        user_prefs["action"],
        user_prefs["comedy"],
        user_prefs["drama"],
        user_prefs["romance"],
        user_prefs["horror"],
        0, 0  # 占位符，用于维度匹配
    ])
    
    # 仅使用类型特征计算相似度
    genre_similarity = cosine_similarity(
        [user_vector[:5]], 
        movie_features[:, :5]
    )[0]
    
    return genre_similarity

# 4. 流行趋势模型
def popularity_based_recommendation(popularity_scores, top_n=5):
    """
    基于流行度的推荐：根据电影流行度进行推荐
    """
    # 归一化流行度分数
    pop_scores = np.array(list(popularity_scores.values()))
    normalized_pop = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())
    return normalized_pop

# 5. 混合推荐系统
def hybrid_recommendation(user_id, alpha=0.7, beta=0.3):
    """
    混合推荐：结合用户偏好和流行趋势
    alpha: 用户偏好权重
    beta: 流行趋势权重
    """
    user_prefs = users_db[user_id]
    
    # 获取基于内容的推荐分数
    content_scores = content_based_recommendation(user_prefs, movie_features, movie_names)
    
    # 获取基于流行度的推荐分数
    pop_scores = popularity_based_recommendation(movie_popularity)
    
    # 组合分数
    hybrid_scores = alpha * content_scores + beta * pop_scores
    
    # 创建结果DataFrame
    recommendations = pd.DataFrame({
        "movie": movie_names,
        "content_score": content_scores,
        "popularity_score": pop_scores,
        "hybrid_score": hybrid_scores
    })
    
    # 排除用户已经看过的电影
    watched_movies = user_history[user_id]
    recommendations = recommendations[~recommendations["movie"].isin(watched_movies)]
    
    # 按混合分数排序
    recommendations = recommendations.sort_values("hybrid_score", ascending=False)
    
    return recommendations

# 6. 产生式规则引擎
def apply_rules(recommendations, user_id):
    """
    应用业务规则优化推荐结果
    """
    user_prefs = users_db[user_id]
    
    # 规则1: 如果用户讨厌恐怖片，降低恐怖片的排名
    if user_prefs["horror"] < 2:
        horror_movies = [movie for movie in recommendations["movie"] 
                         if "horror" in movies_db[movie]["genres"]]
        recommendations.loc[recommendations["movie"].isin(horror_movies), "hybrid_score"] *= 0.5
    
    # 规则2: 如果用户喜欢动作片，提高高评分动作片的排名
    if user_prefs["action"] > 4:
        action_movies = [movie for movie in recommendations["movie"] 
                         if "action" in movies_db[movie]["genres"] and movies_db[movie]["rating"] > 8.5]
        recommendations.loc[recommendations["movie"].isin(action_movies), "hybrid_score"] *= 1.2
    
    # 规则3: 优先推荐近10年的电影
    recent_movies = [movie for movie in recommendations["movie"] 
                    if movies_db[movie]["year"] > 2012]
    recommendations.loc[recommendations["movie"].isin(recent_movies), "hybrid_score"] *= 1.1
    
    # 重新排序
    recommendations = recommendations.sort_values("hybrid_score", ascending=False)
    
    return recommendations

# 7. 为所有用户生成推荐
def generate_all_recommendations():
    all_recommendations = {}
    for user_id in users_db:
        # 获取混合推荐
        rec_df = hybrid_recommendation(user_id)
        
        # 应用业务规则
        rec_df = apply_rules(rec_df, user_id)
        
        # 保存结果
        all_recommendations[user_id] = rec_df.head(10)
    
    return all_recommendations

# 8. 可视化分析
def visualize_recommendations(all_recs):
    # 创建目录保存可视化结果
    viz_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 用户偏好分析
    plt.figure(figsize=(12, 8))
    user_prefs_df = pd.DataFrame(users_db).T
    sns.heatmap(user_prefs_df, annot=True, cmap="YlGnBu")
    plt.title("User Genre Preferences")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "user_preferences.png"))
    plt.close()
    
    # 2. 电影流行度分布
    plt.figure(figsize=(12, 8))
    pop_df = pd.DataFrame(list(movie_popularity.items()), columns=["Movie", "Popularity"])
    pop_df = pop_df.sort_values("Popularity", ascending=False)
    sns.barplot(x="Popularity", y="Movie", data=pop_df, palette="viridis")
    plt.title("Movie Popularity Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "movie_popularity.png"))
    plt.close()
    
    # 3. 推荐分数组成分析
    plt.figure(figsize=(14, 10))
    for i, user_id in enumerate(all_recs):
        user_rec = all_recs[user_id].head(5)
        plt.subplot(3, 2, i+1)
        
        # 创建堆叠条形图
        indices = range(len(user_rec))
        p1 = plt.bar(indices, user_rec["content_score"], width=0.6, label="Content Score")
        p2 = plt.bar(indices, user_rec["popularity_score"], width=0.6, 
                    bottom=user_rec["content_score"], label="Popularity Score")
        
        plt.title(f"{user_id} - Top Recommendations")
        plt.xticks(indices, user_rec["movie"], rotation=45, ha="right")
        plt.ylabel("Score")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "recommendation_composition.png"))
    plt.close()
    
    # 4. 电影类型分布
    genres = ["action", "comedy", "drama", "romance", "horror"]
    genre_counts = {genre: 0 for genre in genres}
    
    for movie in movies_db:
        for genre in movies_db[movie]["genres"]:
            if genre in genre_counts:
                genre_counts[genre] += 1
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(genre_counts.keys()), y=list(genre_counts.values()), palette="muted")
    plt.title("Movie Genre Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "genre_distribution.png"))
    plt.close()

# 9. 保存结果
def save_recommendations(all_recs):
    # 创建目录保存结果
    rec_dir = os.path.join(result_dir, "recommendations")
    os.makedirs(rec_dir, exist_ok=True)
    
    # 保存每个用户的推荐结果
    for user_id, rec_df in all_recs.items():
        rec_df.to_csv(os.path.join(rec_dir, f"{user_id}_recommendations.csv"), index=False)
    
    # 保存电影数据库
    with open(os.path.join(result_dir, "movies_database.json"), "w") as f:
        json.dump(movies_db, f, indent=2)
    
    # 保存用户数据库
    with open(os.path.join(result_dir, "users_database.json"), "w") as f:
        json.dump(users_db, f, indent=2)
    
    # 保存流行度数据
    with open(os.path.join(result_dir, "movie_popularity.json"), "w") as f:
        json.dump(movie_popularity, f, indent=2)

# 10. 主函数
def main():
    print("开始生成电影推荐...")
    
    # 生成所有用户的推荐
    all_recommendations = generate_all_recommendations()
    
    # 保存结果
    save_recommendations(all_recommendations)
    
    # 生成可视化
    visualize_recommendations(all_recommendations)
    
    # 打印示例推荐
    print("\n示例推荐结果:")
    for user_id, rec_df in all_recommendations.items():
        print(f"\n{user_id}的推荐电影:")
        for i, row in rec_df.head(3).iterrows():
            genres = ", ".join(movies_db[row["movie"]]["genres"])
            print(f"- {row['movie']} ({genres}) [综合分数: {row['hybrid_score']:.2f}]")

# 运行主函数
if __name__ == "__main__":
    main()