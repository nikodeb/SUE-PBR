from .user_genres import UserGenresModel
from .bert import BERTModel
from .bert_user_mtl import BERTUserMtlModel

MODELS = {
    BERTModel.code(): BERTModel,
    BERTUserMtlModel.code(): BERTUserMtlModel,
    UserGenresModel.code(): UserGenresModel
}


def model_factory(args):
    """
    Load the specified model
    :param args: system wide arguments from options.py
    :return: architecture
    """
    model = MODELS[args.model_code]
    return model(args)
