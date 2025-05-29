import pandas as pd
from dataset import *
from scipy.sparse import csr_matrix


def ml_load_links():
    df = pd.read_csv(ML_DATASET_PATH + '/links.csv')
    expected_columns = ['movieId', 'imdbId', 'tmdbId']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df


def ml_load_movies():
    df = pd.read_csv(ML_DATASET_PATH + '/movies.csv')
    expected_columns = ['movieId', 'title', 'genres']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df


def ml_load_ratings():
    df = pd.read_csv(ML_DATASET_PATH + '/ratings.csv')
    expected_columns = ['userId', 'movieId', 'rating', 'timestamp']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df


def ml_load_tags():
    df = pd.read_csv(ML_DATASET_PATH + '/tags.csv')
    expected_columns = ['userId', 'movieId', 'tag', 'timestamp']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df


def ml_get_user_item_matrix():
    ratings_df = ml_load_ratings()
    # 使用 pivot_table 构建矩阵，对重复值取平均（可根据需求修改聚合函数）
    matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        aggfunc='mean',  # 若同一用户对同一物品有多次评分，取平均值
        fill_value=0  # 未评分位置填充为0
    )
    return matrix


def ml_get_sparse_user_item_matrix():
    ratings_df = ml_load_ratings()
    """构建稀疏的用户-物品评分矩阵"""
    # 将用户ID和物品ID转换为连续索引
    user_ids = ratings_df['userId'].astype('category')
    item_ids = ratings_df['movieId'].astype('category')

    # 创建稀疏矩阵
    sparse_matrix = csr_matrix(
        (ratings_df['rating'],
         (user_ids.cat.codes, item_ids.cat.codes)),
        shape=(user_ids.cat.categories.size, item_ids.cat.categories.size)
    )

    # 保存索引映射，用于后续查询
    user_id_to_index = dict(zip(user_ids, user_ids.cat.codes))
    item_id_to_index = dict(zip(item_ids, item_ids.cat.codes))

    return sparse_matrix, user_id_to_index, item_id_to_index

def ml_get_user_item_feature():
    """
    构建用户-物品交互的特征矩阵，用于 Factorization Machine 模型训练。
    返回:
        X_sparse: scipy 稀疏矩阵，包含用户与物品的统计/类别特征
        y: ndarray，目标评分值
        user_map: dict，userId 到稀疏矩阵索引的映射
        item_map: dict，movieId 到稀疏矩阵索引的映射
    """
    # 加载基础数据
    ratings_df = ml_load_ratings()
    movies_df = ml_load_movies()
    # tags_df = ml_load_tags()

    # 获取稀疏用户-物品矩阵及映射
    _, user_map, item_map = ml_get_sparse_user_item_matrix()

    # ---------- 用户特征 ----------
    user_stats = ratings_df.groupby('userId').agg(
        rating_count_user=('rating', 'count'),
        avg_rating_user=('rating', 'mean'),
        rating_std_user=('rating', 'std')
    ).reset_index()

    # user_tags = tags_df.fillna({'tag': ''}).groupby('userId')['tag'].apply(
    #     lambda x: str(x)
    # ).reset_index()
    # user_tags.columns = ['userId', 'user_tags']

    # ---------- 物品特征 ----------
    genres_df = movies_df.copy()
    genres_df['genres_list'] = genres_df['genres'].str.split('|')
    genre_dummies = pd.get_dummies(genres_df['genres_list'].explode()).groupby(level=0).sum()
    genres_df = pd.concat([genres_df[['movieId']], genre_dummies], axis=1)

    movie_stats = ratings_df.groupby('movieId').agg(
        rating_count_movie=('rating', 'count'),
        avg_rating_movie=('rating', 'mean'),
        rating_std_movie=('rating', 'std')
    ).reset_index()

    # movie_tags = tags_df.fillna({'tag': ''}).groupby('userId')['tag'].apply(
    #     lambda x: str(x)
    # ).reset_index()
    # movie_tags.columns = ['movieId', 'movie_tags']

    # ---------- 合并所有特征 ----------
    feature_df = ratings_df.copy()

    feature_df = feature_df.merge(user_stats, on='userId', how='left')
    # feature_df = feature_df.merge(user_tags, on='userId', how='left')
    feature_df = feature_df.merge(genres_df, on='movieI0d', how='left')
    feature_df = feature_df.merge(movie_stats, on='movieId', how='left')
    # feature_df = feature_df.merge(movie_tags, on='movieId', how='left')

    # ---------- 缺失值处理 ----------
    feature_df['user_tags'] = feature_df['user_tags'].fillna('')
    feature_df['movie_tags'] = feature_df['movie_tags'].fillna('')
    feature_df = feature_df.fillna(0)

    # ---------- 构造特征矩阵 ----------
    y = feature_df['rating'].values

    numeric_cols = [
        'rating_count_user', 'avg_rating_user', 'rating_std_user',
        'rating_count_movie', 'avg_rating_movie', 'rating_std_movie'
    ]

    genre_cols = genre_dummies.columns.tolist()
    numeric_cols.extend(genre_cols)

    X = feature_df[numeric_cols].values
    X_sparse = csr_matrix(X)

    return X_sparse, y, user_map, item_map


if __name__ == '__main__':
    # sparse_matrix, user_map, item_map = ml_get_sparse_user_item_matrix()
    X_sparse, y, user_map, item_map = ml_get_user_item_feature()
    print(f"稀疏矩阵形状: {X_sparse.shape}")
    print(X_sparse[0])
    # print(f"矩阵密度: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}")
    # 使用示例
    # user_item_matrix = ml_get_user_item_matrix()
    # print(f"评分矩阵形状: {user_item_matrix.shape}")
    # print(user_item_matrix.head())
