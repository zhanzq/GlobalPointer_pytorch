# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com
# date  : 2021/11/15
#

import time
common = {
    "exp_name": "xp_ner",
    "encoder": "BERT",
    "data_home": "./datasets",
    "bert_path": "/Users/zhanzq/Downloads/models/bert-base-chinese",  # bert-base-cased， bert-base-chinese
    "run_type": "train",    # train,eval
    "f1_2_save": 0.5,  # 存模型的最低f1值
    "logger": "default"  # wandb or default，default意味着只输出日志到控制台
}

# wandb的配置，只有在logger=wandb时生效。用于可视化训练过程
wandb_config = {
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    "log_interval": 10
}

train_config = {
    "train_data": "train.json",
    "valid_data": "dev.json",
    "ent2id": "ent2id.json",
    "path_to_save_model": "./outputs",  # 在logger不是wandb时生效
    "hyper_parameters": {
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 50,
        "seed": 2333,
        "max_seq_len": 128,
        "scheduler": "CAWR"
    }
}

eval_config = {
    "model_state_dir": "./outputs/xp_ner/",
    "run_id": "",
    "last_k_model": 1,  # 取倒数第几个model_state
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "save_res_dir": "./results",
    "hyper_parameters": {
        "batch_size": 16,
        "max_seq_len": 512,
    }
    
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------------------------
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}
