from .user_genres import UserGenresTrainer
from .bert import BERTTrainer
from .bert_user_mtl import BERTUserMtlTrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    BERTUserMtlTrainer.code(): BERTUserMtlTrainer,
    UserGenresTrainer.code(): UserGenresTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
