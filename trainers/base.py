import math

from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import multiprocessing


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = self._create_lr_scheduler()

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers, tensorboard_writer=self.writer)
        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch, return_attn=False):
        pass

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch + 1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                if 'accuracy' in self.args.metrics_to_log:
                    description_metrics = ['accuracy']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch + 1,
                'accum_iter': accum_iter,
                'user_embedding': self.model.embedding.user.weight.cpu().detach().numpy()
                if self.args.dump_useritem_embeddings == 'True'
                   and self.model.embedding.user is not None
                else None,
                'item_embedding': self.model.embedding.token.weight.cpu().detach().numpy()
                if self.args.dump_useritem_embeddings == 'True'
                else None,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)

    def test(self):
        print('Test best model with test set!')
        if self.args.save_models_to_disk == 'True':
            if self.args.mode == 'test':
                model_root = self.args.force_load_model_from_location
            else:
                model_root = self.export_root

            best_model = torch.load(os.path.join(model_root, 'models', 'best_acc_model.pth')).get(
                'model_state_dict')
            self.model.load_state_dict(best_model)
            self.model.eval()

            average_meter_set = AverageMeterSet()
            with torch.no_grad():
                tqdm_dataloader = tqdm(self.test_loader)
                for batch_idx, batch in enumerate(tqdm_dataloader):
                    batch = [x.to(self.device) for x in batch]

                    metrics = self.calculate_metrics(batch)

                    for k, v in metrics.items():
                        average_meter_set.update(k, v)
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                          ['Recall@%d' % k for k in self.metric_ks[:3]]
                    if 'accuracy' in self.args.metrics_to_log:
                        description_metrics = ['accuracy']
                    description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    tqdm_dataloader.set_description(description)

                average_metrics = average_meter_set.averages()
                with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                    json.dump(average_metrics, f, indent=4)
                print(average_metrics)

    def _process_metrics_for_csv(self, csv, metric_results_dict, batch, attn_list, layer_1_entropy, input_seq_entropy, scores):
        batch = [x.cpu().detach().numpy() for x in batch]
        scores = scores.cpu().detach().numpy()
        users = None
        if len(batch) == 4:
            seqs, candidates, labels, users = batch
        elif len(batch) == 3:
            seqs, candidates, labels = batch

        assert users.shape[0] == 1

        row = 0
        internal_user_id = users[row][0]
        new_row = [internal_user_id if users is not None else -1]
        new_row += [metric_results_dict['NDCG@%d' % k] for k in self.metric_ks]
        new_row += [metric_results_dict['Recall@%d' % k] for k in self.metric_ks]
        new_row += self._map_internal_movie_list_to_original([int(candidates[0][0])])

        attn_layer_1 = attn_list[0][0]
        attn_layer_2 = attn_list[1][0]

        top_left_coord_to_keep = (attn_layer_1[0] == 0).sum()
        new_row += [200 - top_left_coord_to_keep]

        attn_layer_1 = attn_layer_1[top_left_coord_to_keep:, top_left_coord_to_keep:]
        attn_layer_2 = attn_layer_2[top_left_coord_to_keep:, top_left_coord_to_keep:]

        csv.append(new_row)

        minmax = {
            325: [0.002, 0.03],
            639: [0.018, 0.087],
            616: [0.017, 0.083],
            500: [0.005, 0.044],
            127: [0.015, 0.085],
            115: [0.003, 0.045],
            187: [0.004, 0.045],
            59: [0.004, 0.045],
            627: [0.008, 0.058],
            1094: [0.01, 0.059],

            880: [0.002, 0.029],
            973: [0.005, 0.045],
            1906: [0.017, 0.082],
            1968: [0.001, 0.029],

            226 : [0.013, 0.072],
            490: [0.0175, 0.08],
            1807: [0.012, 0.07],
        }

        minmax_inp_seq = {
            325: [0.0025, 0.018],
            639: [0.021, 0.065],
            616: [0.025, 0.065],
            500: [0.01, 0.038],
            127: [0.02, 0.065],
            115: [0.005, 0.035],
            187: [0.0075, 0.034],
            59: [0.0075, 0.037],
            627: [0.0011, 0.038],
            1094: [0.015, 0.045],

            880: [0.003, 0.0225],
            973: [0.01, 0.039],
            1906: [0.02, 0.06],
            1968: [0.003, 0.025],

            226 : [0.015, 0.057],
            490: [0.02, 0.062],
            1807: [0.013, 0.054],}

        # minmax = {k:[None, None] for k,v in minmax.items()}
        # minmax_inp_seq = {k: [None, None] for k, v in minmax_inp_seq.items()}

        input_item_attn_projection = self._project_attention_on_input(attn_layer_1, attn_layer_2)

        l1_entr = np.average((-attn_layer_1*np.log2(attn_layer_1)).sum(axis=1))
        layer_1_entropy.append(l1_entr)
        inp_entr = (-input_item_attn_projection*np.log2(input_item_attn_projection)).sum(axis=1)[0]
        input_seq_entropy.append(inp_entr)

        return csv, layer_1_entropy, input_seq_entropy
        # if internal_user_id not in minmax:
        #     return csv, layer_1_entropy, input_seq_entropy

        temp_name = 'core_'
        root_dump = os.path.join('Images', 'AttentionTemp', str(internal_user_id))
        # root_dump = os.path.join(self.export_root, 'logs', 'attention', str(internal_user_id))
        Path(root_dump).mkdir(parents=True, exist_ok=True)

        rank = (-scores).argsort(axis=1)
        top10 = candidates[0][rank[0][:10]]

        input_target_dict = {'target': self._map_internal_movie_list_to_original([int(candidates[0][0])]),
                             'predicted': self._map_internal_movie_list_to_original(top10.tolist()),
                             'input_projected_attn': input_item_attn_projection[0].tolist(),
                             'input': self._map_internal_movie_list_to_original([x for x in seqs[row].tolist() if x != 0])}

        with open(os.path.join(root_dump, temp_name+'input_target.json'), 'w') as f:
            json.dump(input_target_dict, f, indent=4)

        min = minmax_inp_seq[internal_user_id][0]
        max = minmax_inp_seq[internal_user_id][1]
        # min, max = None, None
        fig, ax = plt.subplots()
        im = ax.imshow(input_item_attn_projection, cmap='coolwarm', interpolation=None, vmin=min, vmax=max)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attention Weight')
        plt.xlabel('Input Positions')
        plt.yticks([], None)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(os.path.join(root_dump, temp_name+'input_proj.png'), bbox_inches='tight')
        fig.clf()
        plt.close()

        min = minmax[internal_user_id][0]
        max = minmax[internal_user_id][1]
        # min, max = None, None
        fig, ax = plt.subplots()
        im = ax.imshow(attn_layer_1, cmap='coolwarm', interpolation=None, vmin=min, vmax=max)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attention Weight')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(os.path.join(root_dump, temp_name+'layer1.png'), bbox_inches='tight')
        fig.clf()
        plt.close()

        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(attn_layer_2, cmap='coolwarm', interpolation=None)
        cbar2 = ax2.figure.colorbar(im2, ax=ax2)
        cbar2.ax.set_ylabel('Attention Weight')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig2.savefig(os.path.join(root_dump, temp_name+'layer2.png'), bbox_inches='tight')
        fig2.clf()
        plt.close()

        return csv, layer_1_entropy, input_seq_entropy

    def _load_inverse_smap(self):
        self.inverse_smap = {v: k for k, v in self.args.smap.items()}

    def _map_internal_movie_list_to_original(self, list):
        return [self.inverse_smap[id] if id not in [0,1] else id for id in list]

    def _project_attention_on_input(self, layer_1, layer_2):
        pred_item_attn = layer_2[-1][:, None].T
        input_scores = np.dot(pred_item_attn, layer_1)
        return input_scores

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _create_lr_scheduler(self):
        if self.args.lr_sched_type == 'warmup_linear':
            num_epochs = self.args.num_epochs
            num_warmup_steps = self.args.num_warmup_steps

            # Code from huggingface optimisers
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_epochs - current_step) / float(max(1, num_epochs - num_warmup_steps))
                )

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)

        elif self.args.lr_sched_type == 'warmup_cos':
            num_epochs = self.args.num_epochs
            num_warmup_steps = self.args.num_warmup_steps
            num_cycles = 0.5

            # Code from huggingface optimisers
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_epochs - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)

        elif self.args.lr_sched_type == 'cos':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.num_epochs)

        elif self.args.lr_sched_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.decay_step, gamma=self.args.gamma)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        if self.args.save_models_to_disk == 'True':
            # val_loggers.append(RecentModelLogger(model_checkpoint))
            val_loggers.append(BestModelLogger(writer, model_checkpoint, metric_key=self.best_metric))
        val_loggers.append(HparamLogger(writer, args=self.args, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
