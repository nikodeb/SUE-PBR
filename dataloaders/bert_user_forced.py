import random
from random import randrange


from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
from . import utils_forced as utils

import torch
import torch.utils.data as data_utils


class BertUserForcedDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        args.num_users = len(self.umap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
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
        return 'bert_user_forced'

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
        dataset = BertUserMtlTrainDataset(u2seq=self.train,
                                          max_len=self.max_len,
                                          mask_prob=self.mask_prob,
                                          mask_token=self.CLOZE_MASK_TOKEN,
                                          num_items=self.item_count,
                                          rng=self.rng,
                                          bert_force_mask_last=self.args.bert_force_mask_last,
                                          p_window=self.args.bert_p_window,
                                          p_only_mask_last=self.args.bert_p_only_mask_last,
                                          sse_prob=self.args.bert_user_sse_prob,
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
        answers = self.val if mode == 'val' else self.test
        if mode == 'test':
            dataset = BertUserForcedEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                          self.test_negative_samples, val_data=self.val,
                                                s_map=self.smap, u_map=self.umap)
        else:
            dataset = BertUserForcedEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                          self.test_negative_samples, val_data=None,
                                                s_map=self.smap, u_map=self.umap)
        return dataset


class BertUserMtlTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng,
                 bert_force_mask_last=True, p_window=0.5, p_only_mask_last=0.35, sse_prob=0, mask_last_prob=-1.0):
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

        self.sse_prob = sse_prob
        self.sse_keep_prob = 1 - self.sse_prob
        self.sse_uniform_prob = self.sse_prob / (len(self.users) - 1)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        proportion_of_orig_user = 1
        if self.sse_prob > 0:
            new_index = self._get_user_index_sse(index)
            user = self.users[new_index]
            if new_index != index:
                proportion_of_orig_user = 0

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
                        tokens.append(self.rng.randint(2, (self.num_items + 2) - 1))
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

        user_list = [user] * len(tokens)

        return torch.LongTensor(tokens), torch.LongTensor(labels), \
               torch.LongTensor(user_list), torch.FloatTensor([proportion_of_orig_user])

    def _getseq(self, user):
        full_seq = self.u2seq[user]
        if not self.bert_force_last and len(full_seq) > self.max_len:
            sliding_step = (int)(self.p_window * self.max_len) if self.p_window > 0 else self.max_len
            rand_window_start_idx = self.rng.choice(list(range(len(full_seq) - self.max_len, 0, -sliding_step)) + [0])
            return full_seq[rand_window_start_idx: rand_window_start_idx + self.max_len]
        else:
            return full_seq

    def _get_user_index_sse(self, user_index):
        probs = [self.sse_uniform_prob] * len(self.users)
        probs[user_index] = self.sse_keep_prob
        return self.rng.choices(self.users, weights=probs, k=1)[0]


class BertUserForcedEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, val_data=None, s_map=None, u_map=None):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
        self.val_data = val_data

        movies_dir = 'Data/ml-1m/movies.dat'
        users_dir = 'Data/ml-1m/users.dat'
        ratings_dir = 'Data/ml-1m/ratings.dat'
        train = u2seq

        movie_list, users_list, ratings_list = utils.load_text_files([movies_dir, users_dir, ratings_dir])
        self.movie_id_to_name, self.movie_id_to_genres, self.genres_to_movie_ids = utils.process_movie_list(movie_list, s_map)

        self.user_movies_watched, self.user_genres_watched, self.user_genres_count, self.genres_to_users, self.user_genres_percent = \
            utils.process_user_rating_data(train, self.movie_id_to_genres, self.genres_to_movie_ids)

        users_high_horror = utils.get_filtered_users_genre(self.user_genres_percent, genre='Horror', threshold=0.5)
        users_high_comedy = utils.get_filtered_users_genre(self.user_genres_percent, genre='Comedy', threshold=0.5)
        users_high_documentary = utils.get_filtered_users_genre(self.user_genres_percent, genre='Documentary', threshold=0.5)

        u1_h = users_high_horror[2][0]
        u2_d = users_high_documentary[0][0]
        u3_c = users_high_comedy[3][0]

        self.u1_h = u1_h
        self.u2_d = u2_d
        self.u3_c = u3_c

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        if len(answer) > 1:
            answer = answer[-1:]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        no_watched = 10
        total_per_user = 20

        horror_watched_list = list(self.genres_to_movie_ids['Horror'].intersection(self.user_movies_watched[self.u1_h]))
        random.shuffle(horror_watched_list)
        u1_h_watched = horror_watched_list[:no_watched]
        horror_not_watched_list = list(self.genres_to_movie_ids['Horror'].difference(self.user_movies_watched[self.u1_h].union(self.user_movies_watched[self.u2_d])))
        random.shuffle(horror_not_watched_list)
        u1_h_not_watched = horror_not_watched_list[:total_per_user - len(u1_h_watched)]

        comedy_watched_list = list(self.genres_to_movie_ids['Comedy'].intersection(self.user_movies_watched[self.u3_c]))
        random.shuffle(comedy_watched_list)
        u3_c_watched = comedy_watched_list[:no_watched]
        comedy_not_watched_list = list(self.genres_to_movie_ids['Comedy'].difference(self.user_movies_watched[self.u1_h].union(self.user_movies_watched[self.u3_c])))
        random.shuffle(comedy_not_watched_list)
        u3_c_not_watched = comedy_not_watched_list[:total_per_user - len(u3_c_watched)]


        # seq = self.u1_h_watched + self.u1_h_not_watched + self.u2_d_watched + self.u2_d_not_watched
        # seq = self.u1_h_watched + self.u1_h_not_watched + self.u3_c_watched + self.u3_c_not_watched
        seq = u1_h_watched + u1_h_not_watched + u3_c_watched + u3_c_not_watched

        # user_list = [self.u1_h]*len(self.u1_h_watched) + [self.u1_h]*len(self.u1_h_not_watched) + \
        #             [self.u2_d]*len(self.u2_d_watched) + [self.u2_d]*len(self.u2_d_not_watched)
        user_list = [self.u1_h]*len(u1_h_watched) + [self.u1_h]*len(u1_h_not_watched) + \
                    [self.u3_c]*len(u3_c_watched) + [self.u3_c]*len(u3_c_not_watched)
        # print(user_list)

        f = list(self.genres_to_movie_ids['Animation'].difference(self.user_movies_watched[user]))
        random.shuffle(f)
        seq = self.u2seq[user][-20:] + random.sample(range(1,3000), 20)
        user_list = [user]*len(seq)

        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        user_list = [index] * padding_len + user_list

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(
            user_list)
