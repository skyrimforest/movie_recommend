import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class LogisticRegressionRecommender:
    def __init__(self, ratings_df):
        """
        :param ratings_df: 包含 ['userId', 'movieId', 'rating'] 的 DataFrame
        """
        self.ratings_df = ratings_df
        self.model = LogisticRegression()
        self.user_map = {uid: idx for idx, uid in enumerate(ratings_df['userId'].unique())}
        self.item_map = {iid: idx for idx, iid in enumerate(ratings_df['movieId'].unique())}

    def prepare_data(self):
        df = self.ratings_df.copy()
        # 定义隐式反馈标签（打分 >= 4 视为喜欢）
        df['label'] = (df['rating'] >= 4.0).astype(int)

        # 映射为 one-hot 可用的特征（这里只用 ID）
        df['user_idx'] = df['userId'].map(self.user_map)
        df['item_idx'] = df['movieId'].map(self.item_map)

        # 构造特征矩阵
        X = df[['user_idx', 'item_idx']].values
        y = df['label'].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"训练完成，AUC = {auc:.4f}")

    def predict(self, user_id, movie_id):
        if user_id not in self.user_map or movie_id not in self.item_map:
            return 0.0
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[movie_id]
        x = np.array([[user_idx, item_idx]])
        return self.model.predict_proba(x)[0, 1]

    def recommend(self, user_id, N=10):
        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]
        item_indices = list(self.item_map.values())
        inputs = np.array([[user_idx, item_idx] for item_idx in item_indices])
        probs = self.model.predict_proba(inputs)[:, 1]

        top_indices = np.argsort(probs)[::-1][:N]
        top_movie_ids = [list(self.item_map.keys())[i] for i in top_indices]
        top_scores = probs[top_indices]
        return list(zip(top_movie_ids, top_scores))
