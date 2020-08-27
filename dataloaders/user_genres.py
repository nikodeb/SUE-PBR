from .base import AbstractDataloader
import torch
import torch.utils.data as data_utils


class UserGenresDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        """
        This is the data loader used during the favourite genre prediction task
        :param args: system wide arguments from options.py
        :param dataset: the dataset file loaded from disk
        """
        super().__init__(args, dataset)

        self.train = self.dataset['u2g_train']
        self.val = self.dataset['u2g_val']
        self.test = self.dataset['u2g_test']
        self.gmap = self.dataset['gmap']

        args.num_items = len(self.smap)
        args.num_users = len(self.umap)
        args.num_genres = len(self.gmap)

    @classmethod
    def code(cls):
        return 'user_genres'

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
        dataset = UserGenresDataset(u2seq=self.train, rng=self.rng)
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
        # Traditional dataset splitting. Train, val, test all contain mutex users from each other.
        if mode == 'test':
            dataset = UserGenresDataset(u2seq=self.test, rng=self.rng)
        else:
            dataset = UserGenresDataset(u2seq=self.val, rng=self.rng)
        return dataset


class UserGenresDataset(data_utils.Dataset):
    def __init__(self, u2seq, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        genres_pct = self.u2seq[user]

        max_pct = 0.0
        max_genre = None
        for genre, pct in genres_pct.items():
            if pct > max_pct:
                max_pct = pct
                max_genre = genre

        label = max_genre

        return torch.LongTensor([user]), torch.LongTensor([label])