from utils import dataset_loader
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

user_item_matrix: pd.DataFrame
user_sim_df: pd.DataFrame
user_map: dict
item_map: dict

def get_matrix():
    global user_item_matrix,user_map,item_map
    user_item_matrix, user_map, item_map = dataset_loader.ml_get_sparse_user_item_matrix()

def get_user_similarities(user_id):
    global user_item_matrix,user_map,item_map,user_sim_df
    # 获取目标用户的索引
    user_idx =user_map.get(user_id)
    if user_idx is None:
        raise ValueError(f"用户ID {user_id} 不存在")

    # 获取目标用户的评分向量
    user_vector = user_item_matrix[user_idx]

    # 计算余弦相似度（逐行计算，避免生成完整相似度矩阵）
    similarities = []
    for i in range(user_item_matrix.shape[0]):
        sim = cosine_similarity(user_vector, user_item_matrix[i])[0, 0]
        similarities.append(sim)

    # 转换为Series，索引为用户ID
    user_ids = list(user_map.keys())
    user_sim_df = pd.Series(similarities, index=user_ids)

def get_movie_similarities(user_id):
    global user_item_matrix,user_map,item_map,user_sim_df
    # 获取目标用户的索引
    user_idx =user_map.get(user_id)
    if user_idx is None:
        raise ValueError(f"用户ID {user_id} 不存在")

    # 获取目标用户的评分向量
    user_vector = user_item_matrix[user_idx]

    # 计算余弦相似度（逐行计算，避免生成完整相似度矩阵）
    similarities = []
    for i in range(user_item_matrix.shape[0]):
        sim = cosine_similarity(user_vector, user_item_matrix[i])[0, 0]
        similarities.append(sim)

    # 转换为Series，索引为用户ID
    user_ids = list(user_map.keys())
    user_sim_df = pd.Series(similarities, index=user_ids)

def predict_rating(user_id, movie_id, k=5):
    global user_item_matrix,user_sim_df, user_map, item_map

    if user_id not in user_map or movie_id not in item_map:
        return np.nan

    movie_idx = item_map[movie_id]

    # 取该电影的所有用户评分（列向量）
    movie_ratings = user_item_matrix[:, movie_idx].toarray().flatten()

    # 找出所有评分过该电影的用户索引
    rated_users = np.where(movie_ratings > 0)[0]

    # 获取这些用户的相似度
    rated_sims = user_sim_df.iloc[rated_users]

    # 取前 k 个最相似的用户（不包括自己）
    top_k_idx = rated_sims.sort_values(ascending=False).iloc[1:k+1].index
    top_k_sims = rated_sims.loc[top_k_idx]
    top_k_ratings = movie_ratings[top_k_idx]

    # 加权平均评分
    if np.sum(top_k_sims) == 0:
        return np.nan

    predicted_rating = np.dot(top_k_sims, top_k_ratings) / np.sum(top_k_sims)
    return predicted_rating

def recommend_movies(user_id, N=5, k=5):
    global user_item_matrix, user_map, item_map

    if user_id not in user_map:
        raise ValueError(f"用户ID {user_id} 不存在")

    user_idx = user_map[user_id]

    # 获取该用户所有评分记录（行向量）
    user_ratings = user_item_matrix[user_idx].toarray().flatten()

    # 找出未评分的电影索引
    unrated_movie_indices = np.where(user_ratings == 0)[0]

    predictions = {}
    inv_item_map = {v: k for k, v in item_map.items()}

    for idx in unrated_movie_indices:
        movie_id = inv_item_map[idx]
        pred = predict_rating(user_id, movie_id, k=k)
        if not np.isnan(pred):
            predictions[movie_id] = pred

    # 排序后返回前N个
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions[:N]


if __name__ == '__main__':
    get_matrix()
    user_id = 1
    get_user_similarities(user_id)

