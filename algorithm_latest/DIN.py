'''
@Project ：movie_recommend 
@File    ：DIN.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/18 14:02 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_units=[64, 32]):
        super(DIN, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # 注意力网络（计算当前item和历史item之间的相关性）
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 最终预测的MLP（使用加权后的兴趣向量 + 用户向量 + item向量）
        mlp_input_dim = embedding_dim * 3
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
        user_vec = self.user_emb(user_ids)                     # (B, D)
        item_vec = self.item_emb(candidate_items)              # (B, D)
        hist_vecs = self.item_emb(history_items)               # (B, H, D)

        # 扩展维度以便广播
        item_vec_expand = item_vec.unsqueeze(1).expand_as(hist_vecs)  # (B, H, D)

        # attention输入拼接：[hist, item, hist - item, hist * item]
        att_input = torch.cat([
            hist_vecs,
            item_vec_expand,
            hist_vecs - item_vec_expand,
            hist_vecs * item_vec_expand
        ], dim=-1)                                             # (B, H, 4D)

        att_scores = self.attention_mlp(att_input).squeeze(-1) # (B, H)
        att_weights = F.softmax(att_scores, dim=-1)            # (B, H)

        # 用attention加权历史兴趣
        att_weights = att_weights.unsqueeze(-1)                # (B, H, 1)
        weighted_hist = torch.sum(hist_vecs * att_weights, dim=1)  # (B, D)

        # 拼接最终特征向量
        final_input = torch.cat([user_vec, item_vec, weighted_hist], dim=-1)  # (B, 3D)
        out = self.mlp(final_input)                                           # (B, 1)
        return torch.sigmoid(out).squeeze()
