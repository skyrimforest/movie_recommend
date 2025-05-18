'''
@Project ：movie_recommend 
@File    ：SIM.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/18 14:02 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, K=4, hidden_units=[64, 32]):
        super(SIM, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.K = K
        self.embedding_dim = embedding_dim

        # 最后用于点击预测的 MLP
        mlp_input_dim = embedding_dim * 2  # 选中的兴趣 + 目标 item 向量
        layers = []
        dims = [mlp_input_dim] + hidden_units + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, candidate_items, history_items):
        """
        user_ids: (B,)
        candidate_items: (B,)
        history_items: (B, H)
        """
        item_vecs = self.item_emb(history_items)               # (B, H, D)
        cand_vec = self.item_emb(candidate_items)              # (B, D)

        B, H, D = item_vecs.shape
        K = self.K

        # 多兴趣提取：将行为向量聚成 K 类，使用线性变换模拟聚类
        # 简化模拟：用一个线性层将 item_vecs 映射到 K 个兴趣簇
        item_vecs_expand = item_vecs.view(B * H, D)
        interest_proj = nn.Linear(D, K, bias=False).to(item_vecs.device)
        scores = interest_proj(item_vecs_expand).view(B, H, K)  # (B, H, K)

        # Softmax 聚合，得到每个兴趣的权重分布
        scores = F.softmax(scores, dim=1)                      # (B, H, K)

        # 利用权重合并为兴趣向量
        scores = scores.transpose(1, 2)                        # (B, K, H)
        item_vecs = item_vecs.view(B, H, D).unsqueeze(1)      # (B, 1, H, D)
        weighted = torch.matmul(scores.unsqueeze(-2), item_vecs)  # (B, K, 1, D)
        interest_vectors = weighted.squeeze(2)                # (B, K, D)

        # 选出与候选 item 最匹配的一个兴趣向量
        cand_vec_expand = cand_vec.unsqueeze(1).expand(B, K, D)
        sim_scores = torch.sum(interest_vectors * cand_vec_expand, dim=-1)  # (B, K)
        best_interest_idx = torch.argmax(sim_scores, dim=-1)                # (B,)

        selected_interest = interest_vectors[torch.arange(B), best_interest_idx]  # (B, D)

        # 预测点击概率
        final_input = torch.cat([selected_interest, cand_vec], dim=-1)  # (B, 2D)
        out = self.mlp(final_input).squeeze()
        return torch.sigmoid(out)
