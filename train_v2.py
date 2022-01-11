# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/16
#


import sys
from time import localtime, strftime

import logging

from config_v2 import train_config
from transformers import BertTokenizer

import torch

from data_utils_v2 import load_ner_data, compute_ner_metrics, XPNER

from models.gp_v2 import GlobalPointer
from transformers import Trainer, TrainingArguments


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

config = GlobalPointer.config_class(**train_config)
logger.info("config, type: {}, value: {}".format(type(config), config))


def get_param_num(model):
    untrained_param_num, trained_param_num = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            trained_param_num += torch.prod(torch.tensor(p.shape))
        else:
            untrained_param_num += torch.prod(torch.tensor(p.shape))

    return trained_param_num, untrained_param_num


def main():
    log_file = "{}-{}_train.log".format("NER", strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)
    model = GlobalPointer.from_pretrained(config.pretrained_model_dir, config=config)
    logger.info(model)
    trained_param_num, untrained_param_num = get_param_num(model)
    logger.info(msg="trained_param_num: {}, untrained_param_num: {}".format(trained_param_num, untrained_param_num))

    train_data, valid_data, test_data = load_ner_data(config.data_dir, norm_text=True, tokenizer=tokenizer, opt=config)
    train_dataset = XPNER(train_data)
    eval_dataset = XPNER(valid_data)
    test_dataset = XPNER(test_data)

    training_args = TrainingArguments(
        output_dir=config.output_dir,  # output directory
        num_train_epochs=config.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=config.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=config.per_device_eval_batch_size,  # batch size for evaluation
        learning_rate=config.learning_rate,
        adam_epsilon=1e-8,
        warmup_steps=config.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=config.weight_decay,  # strength of weight decay
        logging_dir=config.logging_dir,  # directory for storing logs
        logging_steps=config.logging_steps,
        eval_steps=config.save_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        do_train=True,
        do_eval=True,
        do_predict=True,
        # evaluate_during_training=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy="steps",
        label_names=None,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_ner_metrics,
        tokenizer=tokenizer
    )

    if training_args.do_train:
        trainer.train(config.output_dir)

    if training_args.do_eval:
        trainer.evaluate(eval_dataset=eval_dataset)

    if training_args.do_predict:
        print(trainer.predict(test_dataset=test_dataset).metrics)


if __name__ == "__main__":
    main()
