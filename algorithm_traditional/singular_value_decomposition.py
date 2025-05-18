import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class SVDRecommender:
    def __init__(self, ratings_df, n_components=20):
        """
        ratings_df: 包含 ['userId', 'movieId', 'rating'] 的 DataFrame
        n_components: SVD 分解保留的维度
        """
        self.ratings_df = ratings_df
        self.n_components = n_components
        self.user_map = {uid: idx for idx, uid in enumerate(ratings_df['userId'].unique())}
        self.item_map = {iid: idx for idx, iid in enumerate(ratings_df['movieId'].unique())}
        self.user_map_inv = {v: k for k, v in self.user_map.items()}
        self.item_map_inv = {v: k for k, v in self.item_map.items()}

    def build_matrix(self):
        """
        构造稀疏的用户-物品评分矩阵
        """
        n_users = len(self.user_map)
        n_items = len(self.item_map)

        row = self.ratings_df['userId'].map(self.user_map)
        col = self.ratings_df['movieId'].map(self.item_map)
        data = self.ratings_df['rating']

        self.user_item_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))

    def train(self):
        self.build_matrix()
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd_model = svd.fit(self.user_item_matrix)
        self.reconstructed_matrix = svd.transform(self.user_item_matrix) @ svd.components_

    def predict(self, user_id, movie_id):
        """
        返回用户对电影的预测评分
        """
        if user_id not in self.user_map or movie_id not in self.item_map:
            return np.nan
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[movie_id]
        return self.reconstructed_matrix[user_idx, item_idx]

    def recommend(self, user_id, N=5):
        """
        给用户推荐 N 部未看过的电影
        """
        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        predicted_ratings = self.reconstructed_matrix[user_idx]

        # 找出用户未评分的电影
        unrated_indices = np.where(user_ratings == 0)[0]
        predictions = [(self.item_map_inv[i], predicted_ratings[i]) for i in unrated_indices]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:N]
