from .base import BaseModel
from .bert_modules.bert import BERT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fix_random_seed_as
from models.bert_modules.embedding import BERTEmbedding


class BERTModel(BaseModel):
    def __init__(self, args):
        """
        This is the unpersonalised bidirectional model class
        :param args: system wide parameters from options.py
        """
        super().__init__(args)
        fix_random_seed_as(args.model_init_seed)

        
        self.embedding = BERTEmbedding(vocab_size=args.num_items+2, embed_size=args.bert_hidden_units, max_len=args.bert_max_len, dropout=args.bert_dropout)
        self.bert = BERT(args)
        self.out = nn.Linear(args.bert_hidden_units, args.bert_hidden_units)

        if args.bert_share_in_out_emb:
            # Share the input embedding parameters with the final layer in our recommendation block
            # Therefore we need to create a learned bias vector
            self.vocab_bias = torch.nn.Parameter(nn.init.uniform_(torch.empty(args.num_items + 2),
                                                                  a=-1/math.sqrt(args.num_items + 2),
                                                                  b=1/math.sqrt(args.num_items + 2)),
                                                 requires_grad=True)
        else:
            self.out_final = nn.Linear(args.bert_hidden_units, args.num_items+2)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        mask = x == 0


        x = self.embedding(x)
        x = torch.transpose(x, 0, 1)

        x, attn_list = self.bert(x, mask)
        x = self.out(x)
        x = F.gelu(x)
        if self.args.bert_share_in_out_emb:
            x = torch.matmul(x, self.embedding.token.weight.t())
            x += self.vocab_bias
        else:
            x = self.out_final(x)
        return x, attn_list
