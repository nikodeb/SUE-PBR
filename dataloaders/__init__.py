from .user_genres import UserGenresDataloader
from datasets import dataset_factory
from .bert import BertDataloader
from .bert_user_mtl import BertUserMtlDataloader
from .bert_user_forced import BertUserForcedDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    BertUserMtlDataloader.code(): BertUserMtlDataloader,
    BertUserForcedDataloader.code(): BertUserForcedDataloader,
    UserGenresDataloader.code(): UserGenresDataloader
}


def dataloader_factory(args):
    """
    This method loads the specified dataset using the dataset factory and returns the three data loaders
    :param args: system wide arguments from options.py
    :return: train, validation, test data loaders
    """
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
