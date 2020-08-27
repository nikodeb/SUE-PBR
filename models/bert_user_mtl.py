from .base import BaseModel
from .bert_modules.bert import BERT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fix_random_seed_as
from models.bert_modules.embedding import BERTUserEmbedding

class BERTUserMtlModel(BaseModel):
    def __init__(self, args):
        """
        This is the personalised bidirectional model class.
        It is used for both SSE training and MTL
        :param args: system wide parameters from options.py
        """
        super().__init__(args)

        fix_random_seed_as(args.model_init_seed)

        self.args = args

        
        self.embedding = BERTUserEmbedding(vocab_size=args.num_items+2, embed_size=args.bert_hidden_units,
                                           user_size=args.num_users, user_embed_size=args.bert_user_hidden_units,
                                           max_len=args.bert_max_len, dropout=args.bert_dropout, user_init=args.bert_user_init)
        self.bert = BERT(args)
        self.out = nn.Linear(args.bert_hidden_units + args.bert_user_hidden_units, args.bert_hidden_units - args.bert_user_hidden_units)

        if args.bert_share_in_out_emb:
            # Share the input embedding parameters with the final layer in our recommendation block
            # Therefore we need to create a learned bias vector
            self.vocab_bias = torch.nn.Parameter(nn.init.uniform_(torch.empty(args.num_items + 2),
                                                                  a=-1/math.sqrt(args.num_items + 2),
                                                                  b=1/math.sqrt(args.num_items + 2)),
                                                 requires_grad=True)
        else:
            self.out_final = nn.Linear(args.bert_hidden_units - args.bert_user_hidden_units, args.num_items+2)

        # self.user_hidden_layer = nn.Linear(args.bert_hidden_units + args.bert_user_hidden_units, 64)
        # self.user_final_layer = nn.Linear(64, 1)

        self.user_final_layer = nn.Linear(args.bert_hidden_units + args.bert_user_hidden_units, 1)

        if args.bert_output_context_aggregation == 'user_attn':
            self.bilinear_attn_proj = nn.Linear(args.bert_user_hidden_units, args.bert_hidden_units)
            self.neg_inf_tensor = torch.tensor(float('-inf'), device=args.device)

        if args.bert_dropout_on_userconcat == 'True':
            self.dropout_on_userconcat = nn.Dropout(p=args.bert_p_dropout_on_userconcat)

        if args.bert_dropout_on_mtl == 'True':
            self.dropout_on_mtl = nn.Dropout(p=args.bert_p_dropout_on_mtl)

    @classmethod
    def code(cls):
        return 'bert_user_mtl'

    def forward(self, x, users):
        mask = (x == 0)
        inverse_mask = ~mask
        inverse_mask = inverse_mask.float()
        u_valid_len = torch.sum(inverse_mask, dim=1)


        x, users = self.embedding(x, users)
        x = torch.transpose(x, 0, 1)
        x, attn_list = self.bert(x, mask)

        if self.args.bert_output_context_aggregation == 'user_attn':
            context, user_attn_weights = self.perform_user_attn(x, users, mask)
            u = torch.cat((context, users[:,0,:]), dim=1)
            x = torch.cat((x, users), dim=2)
        elif self.args.bert_output_context_aggregation == 'avgpool':
            x = torch.cat((x, users), dim=2)
            u = x * inverse_mask.unsqueeze(-1).expand_as(x)
            u = torch.sum(u, dim=1)
            u = u / u_valid_len[:,None].expand_as(u)
        else:
            raise NotImplementedError('bert_output_context_aggregation contains invalid option')

        if self.args.bert_dropout_on_userconcat == 'True':
            x = self.dropout_on_userconcat(x)

        x = self.out(x)
        x = F.gelu(x)
        if self.args.bert_share_in_out_emb:
            x = torch.matmul(x, self.embedding.token.weight.t())
            x += self.vocab_bias
        else:
            x = self.out_final(x)

        if self.args.bert_dropout_on_mtl == 'True':
            u = self.dropout_on_mtl(u)
        # u = F.gelu(self.user_hidden_layer(u))
        u = self.user_final_layer(u)

        return x, u, attn_list

    def perform_user_attn(self, bert_out, users, padding_mask):
        users_D = self.bilinear_attn_proj(users[:, 0, :])
        attn_scores = torch.matmul(bert_out, users_D.unsqueeze(-1)).squeeze(-1)
        attn_scores = torch.where(padding_mask, self.neg_inf_tensor, attn_scores)
        attn_weights = F.softmax(attn_scores, dim=1)
        out = torch.matmul(attn_weights.unsqueeze(1), bert_out).squeeze(1)

        return out, attn_weights
