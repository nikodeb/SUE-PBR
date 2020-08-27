from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class BERTUserMtlTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.bce = nn.BCEWithLogitsLoss()

    @classmethod
    def code(cls):
        return 'bert_user_mtl'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels, users, user_targets = batch
        logits, user_pred_logits, attn_list = self.model(seqs, users)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        user_loss = self.bce(user_pred_logits, user_targets)

        if self.args.bert_orig_user_pred_loss_mix >= 0:
            mix = self.args.bert_orig_user_pred_loss_mix
            final_loss = (1-mix)*loss + mix*user_loss
        else:
            mix = -self.args.bert_orig_user_pred_loss_mix
            final_loss = loss + mix*user_loss
        return final_loss

    def calculate_metrics(self, batch, return_attn=False):
        seqs, candidates, labels, users = batch
        scores, user_pred_logits, attn_list = self.model(seqs, users)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        if return_attn:
            return metrics, attn_list, scores
        return metrics
