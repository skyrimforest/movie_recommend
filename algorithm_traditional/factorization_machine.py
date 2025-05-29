import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import dataset_loader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class MatrixFactorization:
    def __init__(self, n_features, n_factors=20, learning_rate=0.01, reg=0.1, n_epochs=20):
        """
        初始化FM模型

        参数:
        - n_features: 特征总数(one-hot编码后的维度)
        """
        self.n_features = n_features
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

        self.user_feature = self.get_user_feature()
        self.item_feature = self.get_item_feature()
        # 初始化参数
        self.w0 = 0.0  # 全局偏置
        self.w = np.zeros(n_features)  # 一阶权重
        self.V = np.random.normal(scale=0.1, size=(n_features, n_factors))  # 二阶交互权重

    def get_user_feature(self):
        self

    def get_item_feature(self):
        self
    def build_fm_features(self, ratings, movies, users):
        """
        构建Factorization Machine的输入特征

        参数:
        - ratings: 评分数据 DataFrame (userId, movieId, rating, timestamp)
        - movies: 电影元数据 DataFrame (movieId, title, genres, year, director, ...)
        - users: 用户元数据 DataFrame (userId, age, gender, occupation, ...)

        返回:
        - X: 特征矩阵
        - y: 目标值(评分)
        """
        # 1. 合并用户、电影和评分数据
        df = ratings.merge(movies, on='movieId').merge(users, on='userId')

        # 2. 提取时间特征
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # 3. 特征分类
        categorical_features = ['userId', 'movieId', 'gender', 'occupation', 'genres', 'day_of_week']
        numerical_features = ['age', 'year', 'hour']
        text_features = ['title']

        # 4. 构建特征处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features),
                ('text', TfidfVectorizer(max_features=100), 'title')
            ])

        # 5. 转换特征
        X = preprocessor.fit_transform(df)
        y = df['rating'].values

        return X, y, preprocessor

    def train(self, X, y, n_epochs=10, batch_size=32):
        """
        模型训练

        参数:
        X: 训练特征矩阵，形状为 [n_samples, n_features]
        y: 目标值，形状为 [n_samples]
        n_epochs: 训练轮数
        batch_size: 批次大小
        """
        n_samples = X.shape[0]

        for epoch in range(n_epochs):
            # 随机打乱样本
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                # 获取当前批次
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # 计算预测值
                y_pred = self.predict(X_batch)

                # 计算误差
                error = y_batch - y_pred

                # 更新偏置项
                self.w0 += self.lr * (np.sum(error) - self.reg * self.w0)

                # 更新一阶权重
                for j in range(self.n_features):
                    if np.sum(X_batch[:, j]) != 0:  # 只更新在当前批次中出现的特征
                        self.w[j] += self.lr * (
                                np.sum(error * X_batch[:, j]) - self.reg * self.w[j]
                        )

                # 更新二阶交互权重
                for j in range(self.n_features):
                    for f in range(self.n_factors):
                        if np.sum(X_batch[:, j]) != 0:
                            # 优化后的梯度计算
                            term = np.dot(X_batch[:, j], error) * np.sum(X_batch[:, :] * self.V[:, f], axis=1)
                            term -= np.sum(error * X_batch[:, j] * X_batch[:, j] * self.V[j, f])

                            self.V[j, f] += self.lr * (term - self.reg * self.V[j, f])

            # 打印每轮的训练误差 (均方误差)
            mse = np.mean(np.square(y - self.predict(X)))
            print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {mse:.6f}")

    def predict(self, X):
        """
        模型预测
        参数:
        X: 输入特征矩阵，形状为 [n_samples, n_features]
        返回:
        y_pred: 预测值，形状为 [n_samples]
        """
        # 一阶项
        linear_term = np.dot(X, self.w)

        # 二阶交互项 (使用优化后的公式)
        interaction_term = 0.5 * np.sum(
            np.power(np.dot(X, self.V), 2) - np.dot(np.power(X, 2), np.power(self.V, 2)),
            axis=1
        )

        # 最终预测
        y_pred = self.w0 + linear_term + interaction_term
        return y_pred

    def recommend_movies_for_user(self, user_id, all_movies_df, user_features_df, preprocessor, top_n=10):
        """
        为指定用户生成电影推荐

        参数:
        - user_id: 目标用户ID
        - all_movies_df: 所有电影的特征DataFrame
        - user_features_df: 所有用户的特征DataFrame
        - preprocessor: 特征预处理管道
        - top_n: 推荐电影数量

        返回:
        - 推荐电影列表（包含电影ID、名称、预测评分等）
        """
        # 获取目标用户的特征
        user_features = user_features_df[user_features_df['userId'] == user_id].copy()

        # 如果用户不存在，返回热门电影（冷启动处理）
        if user_features.empty:
            # return get_popular_movies(all_movies_df, top_n)
            return

        # 为每个电影生成用户-电影对的特征
        user_movie_pairs = []

        for _, movie in all_movies_df.iterrows():
            # 复制用户特征
            pair = user_features.copy()

            # 添加电影特征
            for col in movie.index:
                pair[f'movie_{col}'] = movie[col]

            user_movie_pairs.append(pair)

        # 合并所有用户-电影对
        recommendation_input = pd.concat(user_movie_pairs, ignore_index=True)

        # 应用特征预处理
        X_recommend = preprocessor.transform(recommendation_input)

        # 使用FM模型预测评分
        predicted_ratings = self.predict(X_recommend)

        # 添加预测评分到电影数据
        all_movies_df['predicted_rating'] = predicted_ratings

        # 过滤掉用户已看过的电影（实际应用中需要实现）
        # watched_movies = get_watched_movies(user_id)
        # recommended_movies = all_movies_df[~all_movies_df['movieId'].isin(watched_movies)]

        # 按预测评分排序并返回Top-N
        recommended_movies = all_movies_df.sort_values('predicted_rating', ascending=False).head(top_n)

        return recommended_movies[['movieId', 'title', 'genres', 'predicted_rating']]


if __name__ == '__main__':
    pass
