import os
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import torch


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


def save_embeddings_disk(embedding, path, filename):
    np.save(os.path.join(path, filename), embedding, allow_pickle=True, fix_imports=True)


def save_embeddings_tensorboard(writer, path, filename, tag):
    if os.path.exists(os.path.join(path, filename)):
        embedding = np.load(os.path.join(path, filename), allow_pickle=True, fix_imports=True)
        writer.add_embedding(embedding, metadata=None, tag=tag)


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None, tensorboard_writer=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.tensorboard_writer = tensorboard_writer

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, writer, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_state_dict = None
        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

        self.writer = writer

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            self.best_state_dict = kwargs['state_dict']
            if kwargs['user_embedding'] is not None:
                save_embeddings_disk(kwargs['user_embedding'], self.checkpoint_path, 'user_embedding')
            if kwargs['item_embedding'] is not None:
                save_embeddings_disk(kwargs['item_embedding'], self.checkpoint_path, 'item_embedding')

    def complete(self, *args, **kwargs):
        if self.best_state_dict is not None:
            save_state_dict(self.best_state_dict, self.checkpoint_path, self.filename)

        save_embeddings_tensorboard(self.writer, self.checkpoint_path, 'user_embedding.npy', 'user_embedding')
        save_embeddings_tensorboard(self.writer, self.checkpoint_path, 'item_embedding.npy', 'item_embedding')


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        pass


class HparamLogger(AbstractBaseLogger):
    def __init__(self, writer, args, metric_key='mean_iou'):
        self.writer = writer
        self.args_dict = vars(args)
        self.hparams = args.hparams_to_log
        self.metrics = args.metrics_to_log
        self.metric_key = metric_key
        self.best_metric = 0.

        self.best_metrics_dict = {metric: 0 for metric in self.metrics}
        self.hparams_dict = {hparam: self.args_dict[hparam] for hparam in self.hparams}

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            self.best_metric = current_metric
            for k in self.best_metrics_dict.keys():
                if k in kwargs:
                    self.best_metrics_dict[k] = kwargs[k]

    def complete(self, *args, **kwargs):
        post_processed_dict = {key.replace('@','_'): value for key, value in self.best_metrics_dict.items()}
        self.writer.add_hparams(hparam_dict=self.hparams_dict, metric_dict=post_processed_dict)