# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/12/21
#

# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import argparse
import copy

PARSER_DICT = {
    "ner": [
        {"name": "config", "type": str, "help": "config json file"},
        {"name": "project_dir", "type": str, "help": "project dir"},
        {"name": "data_dir", "type": str, "help": "data dir"},
        {"name": "logging_dir", "type": str, "help": "log dir"},
        {"name": "output_dir", "type": str, "help": "output dir"},
        {"name": "pretrained_model_dir", "type": str, "help": "pretrained model idr"},
        {"name": "model_name", "type": str, "help": "model name", "default": "bert-base-chinese"},
        {"name": "best_model_path", "type": str, "help": "best model path, such as pytorch_model.bin"},
        {"name": "test_data_path", "type": str, "help": "test data path, in jsonl format"},
        {"name": "num_train_epochs", "type": int, "help": "# of training epochs", "default": 5},
        {"name": "per_device_train_batch_size", "type": int, "help": "training batch size per device", "default": 16},
        {"name": "per_device_eval_batch_size", "type": int, "help": "evaluating batch size per device", "default": 64},
        {"name": "learning_rate", "type": float, "help": "learning rate", "default": 5e-5},
        {"name": "max_len", "type": int, "help": "max positional length", "default": 24},
        {"name": "warmup_ratio", "type": float, "help": "% of steps to warm up", "default": 0.1},
        {"name": "warmup_steps", "type": int, "help": "# of steps to warm up", "default": 320},
        {"name": "weight_decay", "type": float, "help": "weight decay", "default": 0.01},
        {"name": "logging_steps", "type": int, "help": "logging steps", "default": 50},
        {"name": "save_steps", "type": int, "help": "save steps", "default": 100},
        {"name": "save_total_limit", "type": int, "help": "# of kept model", "default": 2},
        {"name": "num_labels", "type": int, "help": "# of labels", "default": 6},
        {"name": "gradient_accumulation_steps", "type": int, "help": "# of gradient accumulation steps", "default": 2},
        {"name": "hidden_dropout_prob", "type": float, "help": "hidden dropout prob", "default": 0.05},
        {"name": "attention_probs_dropout_prob", "type": float, "help": "attention dropout prob", "default": 0.05},
        {"name": "use_r_drop", "action": "store_true", "help": "use r-drop or not", "default": None},
        {"name": "ro_pe", "action": "store_true", "help": "use rotated position embedding or not", "default": None},
        {"name": "inner_dim", "type": int, "help": "# of inner dimension in bert", "default": 32},
        {"name": "label2id", "type": dict, "help": "map of label to id",
            "default": {"location": 0, "type": 1, "poiName": 2, "dishName": 3, "taste": 4, "poiIndex": 5}},
        {"name": "id2label", "type": dict, "help": "map of id to label",
            "default": {0: "location", 1: "type", 2: "poiName", 3: "dishName", 4: "taste", 5: "poiIndex"}},
        {"name": "vocab_size", "type": int, "help": "# of vocabulary size", "default": 21128},
        {"name": "device", "type": str, "help": "device type", "default": "cpu"},
    ]
}


def get_parser(parser_name):
    assert parser_name in PARSER_DICT
    parser = argparse.ArgumentParser()
    kwargs_list = copy.deepcopy(PARSER_DICT[parser_name])
    for kwargs in kwargs_list:
        name = kwargs.pop("name", None)
        assert name
        parser.add_argument("--{}".format(name), **kwargs)
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
            for k, v in config.items():
                if hasattr(args, k) and getattr(args, k) is None:
                    setattr(args, k, v)
    return args
