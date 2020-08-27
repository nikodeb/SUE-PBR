from templates import set_template
from datasets import DATASETS
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default=None)

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-20m', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=4, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500, 
                    help='Size of val and test set. 500 for ML-1m and 10000 for ML-20m recommended')

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=int, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert', choices=TRAINERS.keys())
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--enable_lr_schedule', type=bool, default=True, help='Whether to use a LR scheduler')
parser.add_argument('--lr_sched_type', type=str, default='warmup_linear',
                    help='Which LR scheduler to use', choices=['warmup_cos', 'warmup_linear', 'cos', 'step'])
parser.add_argument('--num_warmup_steps', type=int, default=10, help='Number of warmup epochs for a warmup sheduler')
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_user_hidden_units', type=int, default=None, help='Size of user embeddings')
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')
parser.add_argument('--bert_force_mask_last', type=bool, default=False,
                    help='Whether to force only more recent N items to be used')
parser.add_argument('--bert_p_only_mask_last', type=float, default=0.15,
                    help='Probability of only masking last training item in a sequence')
parser.add_argument('--bert_p_window', type=float, default=0.5,
                    help='How many steps to move window relative to maximum input length')
parser.add_argument('--bert_share_in_out_emb', type=bool, default=True,
                    help='Whether to share the input and output embedding matrix')
parser.add_argument('--bert_user_sse_prob', type=float, default=0, help='User SSE-SE probability')
parser.add_argument('--bert_orig_user_pred_loss_mix', type=float, default=0,
                    help='Mixture of original user proportion prediction loss. Value between 0 and 1')
parser.add_argument('--bert_output_context_aggregation', type=str, default='avgpool', choices=['avgpool', 'user_attn'])
parser.add_argument('--bert_dropout_on_userconcat', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--bert_dropout_on_mtl', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--bert_p_dropout_on_userconcat', type=float, default=0.15)
parser.add_argument('--bert_p_dropout_on_mtl', type=float, default=0.15)
parser.add_argument('--bert_mask_last_prob', type=float, default=-1.0)
parser.add_argument('--bert_user_init', type=str, default=None)
################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')
parser.add_argument('--hparams_to_log', nargs='+', type=str, default=['bert_dropout', 'num_epochs', 'lr'])
parser.add_argument('--metrics_to_log', nargs='+', type=str, default=['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10'])
parser.add_argument('--save_models_to_disk', type=str, default='True')
parser.add_argument('--dump_useritem_embeddings', type=str, default='False')
parser.add_argument('--force_load_model_from_location', type=str, default=None)

################
args = parser.parse_args()
set_template(args)
