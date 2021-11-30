"""
Date: 2021-06-11 13:54:00
LastEditors: GodK
LastEditTime: 2021-07-19 21:53:18
"""
import os
import config
import sys
import torch
import json
from data_utils import get_ner_example
from models.gp import GlobalPointer
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Inference:
    """
    A simple inference example
    """
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_model)
        bert = BertModel.from_pretrained(opt.pretrained_bert_model)
        self.model = GlobalPointer(bert, opt).to(opt.device)

        print("loading model {0} from {1}".format(opt.model_name, opt.best_model_path))
        self.model.load_state_dict(torch.load(opt.best_model_path, map_location=opt.device))

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def get_ner(self, text, pred_matrix, threshold=-1.0):
        """
        pred_matrix: shape (1, ner_type_num, seq_len, seq_len)
        """
        id2ent = {val:key for key, val in self.opt.ent2id.items()}
        pred_matrix = pred_matrix.cpu().numpy()

        ent_lst = {}
        for _, ent_type_id, start_idx, end_idx in zip(*np.where(pred_matrix > threshold)):
            ner_type = id2ent[ent_type_id]
            ent_text = text[start_idx-1: end_idx]
            entity = {"value": ent_text, "span": [start_idx-1, end_idx]}
            ent_type_lst = ent_lst.get(ner_type, [])
            ent_type_lst.append(entity)
            ent_lst[ner_type] = ent_type_lst

        return ent_lst


    def evaluate(self, text, threshold=0.0):
        sample = {"text": text}
        data = get_ner_example(sample, norm_text=True, tokenizer=self.tokenizer, opt=self.opt)
        data["input_ids"] = data["input_ids"].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"].unsqueeze(0)
        data["token_type_ids"] = data["token_type_ids"].unsqueeze(0)
        pred_matrix = self.model(data)
        labels = self.get_ner(text, pred_matrix, threshold=threshold)
        predict_res = {"text": text, "label": labels}
        return predict_res


class Option(object):
    def __init__(self, ):
        self.ro_pe = False      # rotated position encoding
        self.dropout = 0.1
        self.inner_dim = 64
        self.max_seq_len = 50
        self.model_name = "GlobalPointer"
        self.best_model_path = "state_dict/gp_xp_ner_val_acc_0.8774_bert-base-chinese_1"
        self.pretrained_bert_model = "/Users/zhanzq/Downloads/models/bert-base-chinese"
        self.ent2id = {"location": 0, "type": 1, "poiName": 2, "dishName": 3, "taste": 4}
        self.ner_type_num = len(self.ent2id)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


def test():
    opt = Option()
    # set your trained models here
    inf = Inference(opt)
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
            labels = inf.evaluate(test_sentence, threshold)
            print(labels)
        except Exception as e:
            print(e)
        print(prompt)
        test_sentence = input()


if __name__ == '__main__':
    test()
