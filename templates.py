def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'
        # args.force_load_model_from_location = 'experiments/core_l2out/chkpts/125_0.2_0.1_0.00005_0.2_2020-06-24_0'
        # args.force_load_model_from_location = 'experiments/core_l2out/chkpts/325_-1.0_0.1_0.0001_0.15_2020-06-24_0'
        # args.force_load_model_from_location = 'experiments/core_l2out/chkpts/325_0.2_0.1_0.00005_0.2_2020-06-24_0'
        args.force_load_model_from_location = 'experiments/core_l2out/chkpts/25_-1.0_0.1_0.0001_0.15_2020-06-24_0'


        code = 'bert'

        args.dataset_code = 'ml-1m'  # + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.save_models_to_disk = 'True'
        args.dump_useritem_embeddings = 'False'
        args.dataloader_code = code
        batch = 64
        seed = 25
        args.dataloader_random_seed = seed
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.dataset_split_seed = seed

        args.train_negative_sampler_code = 'popular'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = seed
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = seed

        args.trainer_code = code
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.weight_decay = 0.0001
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.lr_sched_type = 'warmup_cos'
        args.num_warmup_steps = 20
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = code
        args.model_init_seed = seed

        args.bert_dropout = 0.15
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.4
        args.bert_max_len = 200
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_p_only_mask_last = 0.1
        args.bert_p_window = 0.5
        args.bert_mask_last_prob = -1.0

        args.hparams_to_log = ['model_init_seed', 'bert_dropout']
        args.metrics_to_log = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10']
        args.experiment_dir = 'experiments/randomtests'
        args.experiment_description = 't_core_del'

    elif args.template.startswith('t_train_bert'):
        args.mode = 'train'

        args.dataset_code = 'ml-1m' #+ input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 64
        seed = 25
        args.dataloader_random_seed = seed
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.dataset_split_seed = seed \

        args.train_negative_sampler_code = 'popular'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = seed
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = seed

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.weight_decay = 0.0001
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.lr_sched_type = 'warmup_linear'
        args.num_warmup_steps = 20
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = seed

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.3
        args.bert_max_len = 200
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_p_only_mask_last = 0.1
        args.bert_p_window = 0.5

        args.hparams_to_log = ['model_init_seed', 'lr', 'train_batch_size', 'weight_decay' ]
        args.metrics_to_log = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10' ]
        args.experiment_dir = 'experiments/randomtests'
        args.experiment_description = 'test'

    elif args.template.startswith('u_train_bert'):
        args.mode = 'train'

        code = 'bert_user'

        args.dataset_code = 'ml-1m' #+ input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.save_models_to_disk = 'True'
        args.dump_useritem_embeddings = 'True'
        args.dataloader_code = code
        batch = 64
        seed = 325
        args.dataloader_random_seed = seed
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.dataset_split_seed = seed \

        args.train_negative_sampler_code = 'popular'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = seed
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = seed

        args.trainer_code = code
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.weight_decay = 0.0001
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.lr_sched_type = 'warmup_linear'
        args.num_warmup_steps = 20
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = code
        args.model_init_seed = seed

        args.bert_dropout = 0.2
        args.bert_hidden_units = 320
        args.bert_user_hidden_units = 64
        args.bert_mask_prob = 0.3
        args.bert_max_len = 200
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_p_only_mask_last = 0.1
        args.bert_p_window = 0.5
        args.bert_user_sse_prob = 0.0

        args.hparams_to_log = ['model_init_seed', 'lr', 'train_batch_size', 'bert_user_sse_prob' ]
        args.metrics_to_log = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10' ]
        args.experiment_dir = 'experiments/cleanup/'
        args.experiment_description = 'emb_32064_usernosse'

    elif args.template.startswith('m_train_bert'):
        args.mode = 'train'
        # args.force_load_model_from_location = 'experiments/sse_l2out/chkpts/325_0.4_0.1_-1.0_2020-06-24_0'
        # args.force_load_model_from_location = 'experiments/sse_l2out/chkpts/25_0.4_0.1_-1.0_2020-06-24_0'
        args.force_load_model_from_location = 'experiments/noreg/test/325_0.1_-1.0_2020-07-03_0'

        code = 'bert_user_mtl'

        args.dataset_code = 'ml-1m' #+ input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.save_models_to_disk = 'True'
        args.dump_useritem_embeddings = 'False'
        args.dataloader_code = code
        batch = 64
        seed = 325
        args.dataloader_random_seed = seed
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.dataset_split_seed = seed

        args.train_negative_sampler_code = 'popular'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = seed
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = seed

        args.trainer_code = code
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.weight_decay = 0.0001
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.lr_sched_type = 'warmup_cos'
        args.num_warmup_steps = 20
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = code
        args.model_init_seed = seed

        args.bert_dropout = 0.15
        args.bert_hidden_units = 320
        args.bert_user_hidden_units = 64
        args.bert_mask_prob = 0.4
        args.bert_max_len = 200
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_p_only_mask_last = 0.1
        args.bert_p_window = 0.5
        args.bert_user_sse_prob = 0.3
        args.bert_orig_user_pred_loss_mix = -0.0
        args.bert_output_context_aggregation = 'avgpool'
        args.bert_dropout_on_userconcat = 'True'
        args.bert_dropout_on_mtl = 'True'
        args.bert_p_dropout_on_userconcat = 0.15
        args.bert_p_dropout_on_mtl = 0.15
        args.bert_mask_last_prob = -1.0

        args.hparams_to_log = ['model_init_seed', 'bert_dropout', 'bert_user_sse_prob', 'bert_orig_user_pred_loss_mix' ]
        args.metrics_to_log = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10' ]
        # args.experiment_dir = 'experiments/mtl_wo_hiddenlayer/32064_drp_grp4'
        # args.experiment_description = '325embedding'
        args.experiment_dir = 'experiments/randomtests'
        args.experiment_description = 'sse_afterstatic'
        # args.experiment_dir = 'experiments/randomtests'
        # args.experiment_description = 'tes'

    elif args.template.startswith('x_train_bert'):
        args.mode = 'train'
        args.force_load_model_from_location = 'experiments/mtl_wo_hiddenlayer/32064_drp_grp4/325embedding_2020-06-09_0'

        code = 'bert_user_mtl'

        args.dataset_code = 'ml-1m' #+ input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.save_models_to_disk = 'True'
        args.dump_useritem_embeddings = 'False'
        args.dataloader_code = code
        batch = 64
        seed = 325
        args.dataloader_random_seed = seed
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.dataset_split_seed = seed

        args.train_negative_sampler_code = 'popular'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = seed
        args.test_negative_sampler_code = 'popular'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = seed

        args.trainer_code = code
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.weight_decay = 0.0001
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.lr_sched_type = 'warmup_linear'
        args.num_warmup_steps = 20
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 200 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = code
        args.model_init_seed = seed

        args.bert_dropout = 0.15
        args.bert_hidden_units = 320
        args.bert_user_hidden_units = 64
        args.bert_mask_prob = 0.4
        args.bert_max_len = 200
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.bert_p_only_mask_last = 0.1
        args.bert_p_window = 0.5
        args.bert_user_sse_prob = 0.2
        args.bert_orig_user_pred_loss_mix = -0.05
        args.bert_output_context_aggregation = 'avgpool'
        args.bert_dropout_on_userconcat = 'True'
        args.bert_dropout_on_mtl = 'True'
        args.bert_p_dropout_on_userconcat = 0.15
        args.bert_p_dropout_on_mtl = 0.15

        args.hparams_to_log = ['model_init_seed', 'bert_dropout', 'bert_user_sse_prob', 'bert_orig_user_pred_loss_mix' ]
        args.metrics_to_log = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10' ]
        # args.experiment_dir = 'experiments/mtl_wo_hiddenlayer/32064_drp_grp4'
        # args.experiment_description = '325embedding'
        # args.experiment_dir = 'experiments/mtl_wo_hiddenlayer/32064_drp_grp4'
        # args.experiment_description = 'new2_test_325embedding'
        args.experiment_dir = 'experiments/mtl'
        args.experiment_description = 't_0.2_-0.05'