import pandas as pd
from dataset import *
from scipy.sparse import csr_matrix

def ml_load_links():
    df = pd.read_csv(ML_DATASET_PATH+'/links.csv')
    expected_columns = ['movieId', 'imdbId', 'tmdbId']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df

def ml_load_movies():
    df = pd.read_csv(ML_DATASET_PATH+'/movies.csv')
    expected_columns = ['movieId','title','genres']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df

def ml_load_ratings():
    df = pd.read_csv(ML_DATASET_PATH+'/ratings.csv')
    expected_columns = ['userId','movieId','rating','timestamp']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df

def ml_load_tags():
    df = pd.read_csv(ML_DATASET_PATH+'/tags.csv')
    expected_columns = ['userId','movieId','tag','timestamp']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"文件缺少必要的列。期望列: {expected_columns}")
    return df

def ml_get_title():
    pass

def ml_get_user_item_matrix():
    ratings_df=ml_load_ratings()
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
    ratings_df=ml_load_ratings()
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

if __name__ == '__main__':
    sparse_matrix, user_map, item_map = ml_get_sparse_user_item_matrix()
    print(f"稀疏矩阵形状: {sparse_matrix.shape}")
    print(f"矩阵密度: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}")
    # 使用示例
    # user_item_matrix = ml_get_user_item_matrix()
    # print(f"评分矩阵形状: {user_item_matrix.shape}")
    # print(user_item_matrix.head())
