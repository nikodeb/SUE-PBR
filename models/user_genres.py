from .base import BaseModel
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import fix_random_seed_as
from models.bert_modules.embedding.user import UserEmbedding

class UserGenresModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        fix_random_seed_as(args.model_init_seed)

        self.args = args


        self.embedding = UserEmbedding(user_size=args.num_users, embed_size=args.bert_user_hidden_units)
        self.load_pretrained_embeddings()
        self.embedding.weight.requires_grad=False


        self.linear1 = nn.Linear(args.bert_user_hidden_units, 64*4, bias=True)
        self.linear2 = nn.Linear(64*4, 64, bias=True)
        # self.out = nn.Linear(64, 18, bias=True)

        self.out = nn.Linear(64, 18, bias=True)
        self.dropout = nn.Dropout(args.bert_dropout)

    @classmethod
    def code(cls):
        return 'user_genres'

    def load_pretrained_embeddings(self):
        embedding_matrix = np.load(self.args.user_embedding_path)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, users):
        users = users.squeeze(-1)
        x = self.embedding(users)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x