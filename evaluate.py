# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com
# date  : 2021/12/07
#

import os
import json
from time import localtime, strftime

import torch
import numpy

import sys

from parser import get_parser
from transformers import BertTokenizer, TrainingArguments, Trainer
from models.gp_v2 import GlobalPointer
from data_utils_v2 import get_ner_example, XPNER, load_ner_data, compute_ner_metrics

from utils import save_to_jsonl, get_default_params

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Inference:
    """
    A simple inference example
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)
        self.model = GlobalPointer(config=config).to(config.device)

        self.print_args()

        logger.info("loading best NER model from {}".format(config.best_model_path))
        self.model.load_state_dict(torch.load(config.best_model_path, map_location=config.device))

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

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

    def get_entities(self, text, pred_matrix, threshold=0.0):
        """
        get all entities in text, with threshold
        :param text:
        :param pred_matrix: shape (1, ner_type_num, seq_len, seq_len)
        :param threshold: default 0.0
        :return: {"type1": [{"value": "val1", "pos": [start1, end1]}, {}], "type2": []}
        """
        id2label = self.config.id2label
        pred_matrix = pred_matrix.cpu().numpy()

        entities = {}
        for _, ent_type_id, start_idx, end_idx in zip(*numpy.where(pred_matrix > threshold)):
            entity_type = id2label[ent_type_id]
            entity_name = text[start_idx-1: end_idx]
            entity = {"value": entity_name, "pos": [start_idx-1, end_idx]}
            ent_type_lst = entities.get(entity_type, [])
            ent_type_lst.append(entity)
            entities[entity_type] = ent_type_lst

        return entities

    def infer(self, text, threshold=0.0):
        sample = {"text": text}
        data = get_ner_example(sample, norm_text=True, tokenizer=self.tokenizer, opt=self.config)
        data["input_ids"] = data["input_ids"].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"].unsqueeze(0)
        data["token_type_ids"] = data["token_type_ids"].unsqueeze(0)
        pred_matrix = self.model(data["input_ids"], data["attention_mask"], data["token_type_ids"])[0]
        labels = self.get_entities(data["sample"]["text"], pred_matrix, threshold=threshold)
        predict_res = {"text": text, "label": labels}

        return predict_res

    def load_data(self):
        train_data, valid_data, test_data = load_ner_data(self.config.data_dir,
                                                          norm_text=True,
                                                          tokenizer=self.tokenizer,
                                                          opt=self.config)
        train_dataset = XPNER(train_data)
        eval_dataset = XPNER(valid_data)
        test_dataset = XPNER(test_data)

        return train_dataset, eval_dataset, test_dataset

    def evaluate_on_dataset(self,):
        train_dataset, eval_dataset, test_dataset = self.load_data()
        config_dct = self.config.__dict__
        config_dct = get_default_params(config_dct, TrainingArguments)
        training_args = TrainingArguments(**config_dct)

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=eval_dataset,  # evaluation dataset
            compute_metrics=compute_ner_metrics,
            tokenizer=self.tokenizer
        )

        self._evaluate_on_dataset(trainer=trainer, dataset=train_dataset, tag="train")
        self._evaluate_on_dataset(trainer=trainer, dataset=eval_dataset, tag="eval")
        self._evaluate_on_dataset(trainer=trainer, dataset=test_dataset, tag="test")

    @staticmethod
    def _evaluate_on_dataset(trainer, dataset, tag="eval"):
        outputs = trainer.predict(test_dataset=dataset)
        test_metrics = {"{}_{}".format(tag, key[5:]): val for key, val in outputs.metrics.items()}
        logger.info(test_metrics)

    def pred_on_dataset(self, data_path=None, output_dir=None, threshold=0.0):
        if not data_path:
            data_path = self.config.test_data_path
        if not output_dir:
            output_dir = self.config.output_dir
        diffs, preds = [], []
        with open(data_path, "r", encoding="utf-8") as reader:
            for line in reader:
                sample = json.loads(line.strip())
                pred = self.infer(sample["text"], threshold)
                diff = get_diff(sample, pred)
                if diff is not None:
                    diffs.append(diff)
                preds.append(sample)

        pred_path = os.path.join(output_dir, "pred_{}_{}".format(threshold, data_path.split("/")[-1]))
        save_to_jsonl(json_lst=preds, jsonl_path=pred_path)

        diff_path = os.path.join(output_dir, "diff_{}_{}".format(threshold, data_path.split("/")[-1]))
        save_to_jsonl(json_lst=diffs, jsonl_path=diff_path)

    def interactive_test(self,):
        prompt = "请输入用户语句:"
        print(prompt)
        threshold = 0.0
        test_sentence = input()
        while test_sentence:
            try:
                while test_sentence.lower() == "set":
                    print("请输入NER阈值:")
                    threshold = float(input())
                    print(prompt)
                    test_sentence = input()
                labels = self.infer(test_sentence, threshold)
                print(labels)
            except Exception as e:
                print(e)
            print(prompt)
            test_sentence = input()


def convert_dict_to_list(dct):
    lst = []
    for key in dct:
        val_lst = dct[key]
        for item in val_lst:
            lst.append([key, item["value"], item["pos"]])
    lst.sort(key=lambda it: it[-1])
    lst = [[it[0], it[1]] for it in lst]

    return lst


def get_diff(sample, pred):
    text = sample["text"]
    label = sample["label"]
    label2 = convert_dict_to_list(pred["label"])
    sample["pred"] = label2
    if len(label) == len(label2):
        label.sort()
        tmp_label2 = sorted(label2)
        if label == tmp_label2:
            sample["label"] = label2
            return None

    return {"text": text, "label": label, "pred": label2}


def main():
    log_file = "{}-{}_evaluation.log".format("NER", strftime("%y%m%d-%H%M", localtime()))
    problem = sys.argv.pop(1)
    args = get_parser(problem)
    log_file = os.path.join(args.logging_dir, log_file)
    logger.addHandler(logging.FileHandler(log_file))

    args.id2label = {val: key for key, val in args.label2id.items()}
    config = GlobalPointer.config_class(**vars(args))

    inf = Inference(config=config)

    inf.evaluate_on_dataset()
    inf.pred_on_dataset()
    # inf.interactive_test()


if __name__ == "__main__":
    main()
