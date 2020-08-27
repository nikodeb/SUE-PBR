from .ml_1m import ML1MDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset
}


def dataset_factory(args):
    """
    Load the specified dataset object
    :param args: system wide arguments from options.py
    :return: dataset object
    """
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
