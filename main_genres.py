import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def setup_model_args(args, seed, batch_size, learn_rate, weight_decay, dropout, epochs):
    model_args = args
    model_args.mode = 'train'
    model_args.force_load_model_from_location = 'experiments/sse_l2out/chkpts/25_0.4_0.1_-1.0_2020-06-24_0'

    code = 'user_genres'

    model_args.dataset_code = 'ml-1m'
    model_args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
    model_args.min_uc = 5
    model_args.min_sc = 0
    model_args.split = 'leave_one_out'

    model_args.save_models_to_disk = 'True'
    model_args.dump_useritem_embeddings = 'False'
    model_args.dataloader_code = code
    batch = batch_size

    model_args.dataloader_random_seed = seed
    model_args.train_batch_size = batch
    model_args.val_batch_size = batch
    model_args.test_batch_size = batch
    model_args.dataset_split_seed = seed

    model_args.train_negative_sampler_code = 'popular'
    model_args.train_negative_sample_size = 0
    model_args.train_negative_sampling_seed = seed
    model_args.test_negative_sampler_code = 'popular'
    model_args.test_negative_sample_size = 100
    model_args.test_negative_sampling_seed = seed

    model_args.trainer_code = code
    model_args.device = 'cuda'
    model_args.num_gpu = 1
    model_args.device_idx = '0'
    model_args.optimizer = 'Adam'
    model_args.weight_decay = weight_decay
    model_args.lr = learn_rate
    model_args.enable_lr_schedule = True
    model_args.lr_sched_type = 'warmup_cos'
    model_args.num_warmup_steps = 0
    model_args.decay_step = 25
    model_args.num_epochs = epochs
    model_args.best_metric = 'accuracy'

    model_args.model_code = code
    model_args.model_init_seed = seed

    model_args.bert_dropout = dropout
    model_args.bert_hidden_units = 320
    model_args.bert_user_hidden_units = 64

    # model_args.user_embedding_path = 'experiments/sse_np_2020-07-10_0/models/user_embedding.npy'
    model_args.user_embedding_path = 'experiments/noreg_np_2020-07-30_0/models/user_embedding.npy'

    model_args.hparams_to_log = ['model_init_seed', 'train_batch_size', 'lr', 'weight_decay', 'bert_dropout', 'num_epochs']
    model_args.metrics_to_log = ['accuracy']
    model_args.experiment_dir = 'experiments/noreg_genres_mlp'
    model_args.experiment_description = '{}_{}_{}_{}_{}_{}'.format(seed, batch_size, learn_rate, weight_decay, dropout, epochs)
    return model_args


def train(model_args):
    export_root = setup_train(model_args)
    train_loader, val_loader, test_loader = dataloader_factory(model_args)
    model = model_factory(model_args)
    trainer = trainer_factory(model_args, model, train_loader, val_loader, test_loader, export_root)
    if model_args.mode == 'train':
        trainer.train()
    trainer.test()


if __name__ == '__main__':
    bss = [
           # 500,
           # 1000,
           2000
           ]
    lrs = [0.01, 0.001]
    wds = [0.00001, 0.000001]
    drs = [0.1, 0.2, 0.3]

    for batch_size in bss:
        for learning_rate in lrs:
            for weight_decay in wds:
                for dropout in drs:
                    seed = 25
                    epochs = 100

                    print('\n\n')
                    print('PARAMS: {}_{}_{}_{}_{}_{}'.format(seed, batch_size, learning_rate, weight_decay, dropout, epochs))
                    model_args = setup_model_args(args,
                                                  seed=seed,
                                                  batch_size=batch_size,
                                                  learn_rate=learning_rate,
                                                  weight_decay=weight_decay,
                                                  dropout=dropout,
                                                  epochs=epochs)
                    train(model_args)