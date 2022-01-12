# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/16
#

import sys
import torch
from time import localtime, strftime

from models.gp_v2 import GlobalPointer
from transformers import Trainer, TrainingArguments, BertTokenizer
from data_utils_v2 import load_ner_data, compute_ner_metrics, XPNER

import logging

from parser import get_parser

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    """
    A simple inference example
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)
        logger.info("loading pretrained NER model from {}".format(config.pretrained_model_dir))
        self.model = GlobalPointer.from_pretrained(config.pretrained_model_dir, config=config).to(config.device)

        self.print_args()

        # load dataset
        logger.info("loading dataset ...")
        self.train_dataset, self.eval_dataset, self.test_dataset = self.load_data()

    def print_args(self):
        n_trainable_params, n_untrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_untrainable_params += n_params
        logger.info("> n_trainable_params: %d, n_untrainable_params: %d" % (n_trainable_params, n_untrainable_params))
        logger.info("> training arguments:")
        for arg in vars(self.config):
            logger.info(">>> {0}: {1}".format(arg, getattr(self.config, arg)))

    def load_data(self):
        train_data, valid_data, test_data = load_ner_data(self.config.data_dir,
                                                          norm_text=True,
                                                          tokenizer=self.tokenizer,
                                                          opt=self.config)
        train_dataset = XPNER(train_data)
        eval_dataset = XPNER(valid_data)
        test_dataset = XPNER(test_data)

        return train_dataset, eval_dataset, test_dataset

    def get_trainer(self):
        config_dct = self.config.__dict__
        training_args = TrainingArguments(**config_dct)

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.eval_dataset,  # evaluation dataset
            compute_metrics=compute_ner_metrics,
            tokenizer=self.tokenizer
        )

        return trainer

    def train(self):
        self.config.do_train = True
        self.config.do_eval = True
        self.config.do_predict = False
        trainer = self.get_trainer()
        trainer.train(self.config.output_dir)

    def evaluate(self):
        self.config.do_train = False
        self.config.do_eval = True
        self.config.do_predict = False
        logger.info("loading best NER model from {}".format(self.config.best_model_path))
        self.model.load_state_dict(torch.load(self.config.best_model_path, map_location=self.config.device))
        trainer = self.get_trainer()
        trainer.evaluate(eval_dataset=self.eval_dataset)

    def predict(self):
        self.config.do_train = False
        self.config.do_eval = False
        self.config.do_predict = True
        logger.info("loading best NER model from {}".format(self.config.best_model_path))
        self.model.load_state_dict(torch.load(self.config.best_model_path, map_location=self.config.device))
        trainer = self.get_trainer()
        test_res = trainer.predict(test_dataset=self.test_dataset).metrics
        logger.info(test_res)


def main():
    log_file = "{}-{}_train.log".format("NER", strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    problem = sys.argv.pop(1)
    args = get_parser(problem)
    args.id2label = {val: key for key, val in args.label2id.items()}
    config = GlobalPointer.config_class(**vars(args))

    ins = Instructor(config=config)

    if config.do_train:
        ins.train()

    if config.do_eval:
        ins.evaluate()

    if config.do_predict:
        ins.predict()


if __name__ == "__main__":
    main()
