import torch
import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .user import UserEmbedding
import pickle
import numpy as np

class BERTUserEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, user_size, user_embed_size, max_len, dropout=0.1, user_init=None):

        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size - user_embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.user = UserEmbedding(user_size=user_size, embed_size=user_embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        if user_init is not None:
            if user_init == 'ones':
                nn.init.ones_(self.user.weight.data)
            elif user_init == 'zeroes':
                nn.init.zeros_(self.user.weight.data)
            elif user_init == 'standard_indep':
                stats = pickle.load(open("u_emb_stats.p", "rb"))
                mean = torch.from_numpy(stats['mean'])
                std = torch.from_numpy(np.sqrt(stats['var']))
                self.user.weight.data = torch.distributions.normal.Normal(mean, std).sample(torch.Size([user_size]))
            elif user_init == 'standard_cov':
                stats = pickle.load(open("u_emb_stats.p", "rb"))
                mean = torch.from_numpy(stats['mean'])
                cov = torch.from_numpy(stats['cov']).float()
                self.user.weight.data = torch.distributions.multivariate_normal\
                    .MultivariateNormal(loc=mean, covariance_matrix=cov)\
                    .sample(torch.Size([user_size]))
            self.user.weight.requires_grad = False

    def forward(self, sequence, user):
        token_embs = self.token(sequence)
        user_embs = self.user(user)
        x = torch.cat((token_embs, user_embs), dim=2) + self.position(sequence)
        return self.dropout(x), user_embs
