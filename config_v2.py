# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com
# date  : 2021/11/15
#

import time

train_config = {
    "model_name": "mengzi-base-chinese",
    "pretrained_model_dir": "/Users/zhanzq/Downloads/models/",
    "project_dir": "/Users/zhanzq/github/GlobalPointer_pytorch/",
    "data_dir": "./datasets/xp_ner_1129/",
    "output_dir": "./output/",
    "logging_dir": "./log/",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "learning_rate": 5e-5,
    "max_len": 24,
    "warmup_ratio": 0.1,
    "warmup_steps": 320,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "save_steps": 100,
    "save_total_limit": 2,
    "num_labels": 6,
    "gradient_accumulation_steps": 2,
    "hidden_dropout_prob": 0.05,
    "attention_probs_dropout_prob": 0.05,
    "use_r_drop": False,
    "ro_pe": False,
    "inner_dim": 32,
    "vocab_size": 21128,
}

local = True
if local:
    project_dir = "/Users/zhanzq/github/GlobalPointer_pytorch/"
    pretrained_model_dir = "/Users/zhanzq/Downloads/models/"
else:
    project_dir = "/Users/zhanzq/github/GlobalPointer_pytorch/"
    pretrained_model_dir = "/Users/zhanzq/Downloads/models/"

train_config["pretrained_model_path"] = os.path.join(pretrained_model_dir, train_config["model_name"])
train_config["data_dir"] = os.path.join(project_dir, train_config["data_dir"])
train_config["output_dir"] = os.path.join(project_dir, train_config["output_dir"])
train_config["logging_dir"] = os.path.join(project_dir, train_config["logging_dir"])

label2id = {"location": 0, "type": 1, "poiName": 2, "dishName": 3, "taste": 4}
id2label = {val:key for key,val in label2id.items()}
train_config["label2id"] = label2id
train_config["id2label"] = id2label

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

# ---------------------------------------------
train_config = {**train_config}
eval_config = {**eval_config}
