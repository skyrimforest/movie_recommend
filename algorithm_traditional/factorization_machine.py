import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MatrixFactorization:
    def __init__(self, ratings, n_factors=20, learning_rate=0.01, reg=0.1, n_epochs=20):
        """
        :param ratings: pd.DataFrame, 包含 userId, movieId, rating 列
        """
        self.ratings = ratings
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

        self.user_map = {uid: idx for idx, uid in enumerate(ratings['userId'].unique())}
        self.item_map = {iid: idx for idx, iid in enumerate(ratings['movieId'].unique())}
        self.user_inv_map = {v: k for k, v in self.user_map.items()}
        self.item_inv_map = {v: k for k, v in self.item_map.items()}

        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)

        # 初始化因子矩阵
        self.P = np.random.normal(scale=0.1, size=(self.n_users, self.n_factors))  # 用户矩阵
        self.Q = np.random.normal(scale=0.1, size=(self.n_items, self.n_factors))  # 物品矩阵

    def train(self):
        for epoch in range(self.n_epochs):
            loss = 0
            for _, row in self.ratings.iterrows():
                u = self.user_map[row['userId']]
                i = self.item_map[row['movieId']]
                r_ui = row['rating']

                pred = np.dot(self.P[u], self.Q[i])
                err = r_ui - pred
                loss += err ** 2

                # SGD 更新
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])

            print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss = {loss:.4f}")

    def predict(self, user_id, movie_id):
        if user_id not in self.user_map or movie_id not in self.item_map:
            return np.nan
        u = self.user_map[user_id]
        i = self.item_map[movie_id]
        return np.dot(self.P[u], self.Q[i])

    def recommend(self, user_id, N=10):
        if user_id not in self.user_map:
            return []
        u = self.user_map[user_id]
        scores = np.dot(self.Q, self.P[u])
        movie_ids = np.array([self.item_inv_map[i] for i in range(self.n_items)])
        recommendations = sorted(zip(movie_ids, scores), key=lambda x: x[1], reverse=True)
        return recommendations[:N]
