import torch

from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class UserGenresTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss()

    @classmethod
    def code(cls):
        return 'user_genres'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        users, labels = batch
        logits = self.model(users)
        labels = labels.squeeze(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, return_attn=False):
        users, labels = batch
        labels = labels.squeeze(-1)
        logits = self.model(users)
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / labels.shape[0]
        return {'accuracy': accuracy}
