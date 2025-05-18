import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

class GBDTRecommender:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df.copy()
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def preprocess(self):
        # 编码 userId 和 movieId 为连续整数
        self.ratings_df['user'] = self.user_encoder.fit_transform(self.ratings_df['userId'])
        self.ratings_df['item'] = self.item_encoder.fit_transform(self.ratings_df['movieId'])

    def train(self):
        self.preprocess()
        X = self.ratings_df[['user', 'item']]
        y = self.ratings_df['rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=10, verbose=False)

        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"✅ GBDT 模型训练完成，验证 RMSE = {rmse:.4f}")

    def predict(self, user_id, movie_id):
        user_enc = self.user_encoder.transform([user_id])[0] if user_id in self.user_encoder.classes_ else -1
        item_enc = self.item_encoder.transform([movie_id])[0] if movie_id in self.item_encoder.classes_ else -1

        if user_enc == -1 or item_enc == -1:
            return np.nan
        return self.model.predict([[user_enc, item_enc]])[0]

    def recommend(self, user_id, candidate_movie_ids, N=5):
        if user_id not in self.user_encoder.classes_:
            return []

        user_enc = self.user_encoder.transform([user_id])[0]
        items_enc = self.item_encoder.transform(candidate_movie_ids)

        X_pred = pd.DataFrame({
            'user': [user_enc] * len(items_enc),
            'item': items_enc
        })

        preds = self.model.predict(X_pred)
        results = list(zip(candidate_movie_ids, preds))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:N]
