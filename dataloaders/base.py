from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        """
        This is the base data loader class shared between all other loaders.
        :param args: contains all the program arguments from options.py
        :param dataset: the dataset object loaded from disk
        """
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.dataset = dataset
        args.umap = dataset['umap']
        args.smap = dataset['smap']
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        """
        :return: the class code which is used to select the correct subclass using args
        """
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        """
        :return: the dataloaders specified
        """
        pass
