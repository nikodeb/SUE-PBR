import torch.nn as nn


class UserEmbedding(nn.Embedding):
    def __init__(self, user_size, embed_size=512):
        super().__init__(user_size, embed_size)
        # User embedding indexes start at 0
        # 0 is not a padding index!
