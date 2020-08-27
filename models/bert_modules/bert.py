from torch import nn as nn
import torch
from .transformer import TransformerEncoderLayer

class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()


        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*4,
                                           dropout=dropout, activation='gelu') for _ in range(n_layers)])

    def forward(self, x, mask):

        attn_list = []
        
        for transformer in self.transformer_blocks:
            x, attn = transformer.forward(src=x, src_mask=None, src_key_padding_mask=mask)
            attn_list.append(attn.detach().cpu().numpy())
        x = torch.transpose(x, 0, 1)

        return x, attn_list

    def init_weights(self):
        pass
