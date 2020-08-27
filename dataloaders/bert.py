from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        """
        This is the data loader class for the unpersonalised bidirectional model.
        :param args: contains all the program arguments from options.py
        :param dataset: the dataset object loaded from disk
        """
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        args.num_users = len(self.umap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob

        # [MASK] token has ID of 1, padding ID 0
        self.CLOZE_MASK_TOKEN = 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=0)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob
                                   , self.CLOZE_MASK_TOKEN, self.item_count,
                                   self.rng, self.args.bert_force_mask_last,
                                   self.args.bert_p_window, self.args.bert_p_only_mask_last,
                                   mask_last_prob=self.args.bert_mask_last_prob)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True, num_workers=0)
        return dataloader

    def _get_eval_dataset(self, mode):
        # For the validation dataset, simply use train + validation, val used for metrics.
        # For the test dataset, use train + validation + test, test used for metrics
        answers = self.val if mode == 'val' else self.test
        if mode == 'test':
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                          self.test_negative_samples, val_data=self.val)
        else:
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                          self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng, bert_force_mask_last=True, p_window=0.5,
                 p_only_mask_last=0.35, mask_last_prob=-1.0):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_last_prob = mask_last_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.bert_force_last = bert_force_mask_last
        self.p_window = p_window
        self.p_only_mask_last = p_only_mask_last

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []

        if self.rng.random() < self.p_only_mask_last:
            tokens = [s for s in seq]
            tokens[-1] = self.mask_token
            labels = [0] * len(seq)
            labels[-1] = seq[-1]
        else:
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(2, (self.num_items+2)-1))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            if self.mask_last_prob >= 0:
                if self.rng.random() < self.mask_last_prob:
                    labels[-1] = seq[-1]
                    tokens[-1] = self.mask_token
                else:
                    labels[-1] = 0
                    tokens[-1] = seq[-1]

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        # Windowing strategy. Step size of p_window * max_len

        full_seq = self.u2seq[user]
        if not self.bert_force_last and len(full_seq) > self.max_len:
            sliding_step = (int)(self.p_window * self.max_len) if self.p_window > 0 else self.max_len
            rand_window_start_idx = self.rng.choice(list(range(len(full_seq)-self.max_len, 0, -sliding_step))+[0])
            return full_seq[rand_window_start_idx: rand_window_start_idx + self.max_len]
        else:
            return full_seq



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, val_data=None):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
        self.val_data = val_data

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        if len(answer) > 1:
            seq = seq + answer[:-1]
            answer = answer[-1:]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        if self.val_data is not None:
            seq = seq + self.val_data[user]

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        user_list = [user] * len(seq)

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(user_list)

