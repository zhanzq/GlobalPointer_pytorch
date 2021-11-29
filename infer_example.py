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
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Inference:
    """
    A simple inference example
    """
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = BertTokenizerFast.from_pretrained(opt.pretrained_bert_model)
        bert = BertModel.from_pretrained(opt.pretrained_bert_model)
        self.model = GlobalPointer(bert, opt).to(opt.device)

        print("loading model {0} from {}".format(opt.model_name, opt.best_model_path))
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(opt.best_model_path))
        else:
            self.model.load_state_dict(torch.load(opt.best_model_path, map_location=torch.device("cpu")))

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def decode_ent(self, text, pred_matrix, tokenizer, threshold=0):
        # print(text)
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        id2ent = {id: ent for ent, id in self.opt.ent2id.items()}
        pred_matrix = pred_matrix.cpu().numpy()
        ent_list = {}
        for ent_type_id, token_start_index, toekn_end_index in zip(*np.where(pred_matrix > threshold)):
            ent_type = id2ent[ent_type_id]
            ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[toekn_end_index][1]]
            ent_text = text[ent_char_span[0]:ent_char_span[1]]

            ent_type_dict = ent_list.get(ent_type, {})
            ent_text_list = ent_type_dict.get(ent_text, [])
            ent_text_list.append(ent_char_span)
            ent_type_dict.update({ent_text: ent_text_list})
            ent_list.update({ent_type:ent_type_dict})
        # print(ent_list)
        return ent_list

    def evaluate(self, text):
        sample = {"text": text}
        data = get_ner_example(sample, norm_text=True, tokenizer=self.tokenizer, opt=self.opt)
        pred_matrix = self.model(data)
        labels = self.decode_ent(text, pred_matrix, self.tokenizer)
        predict_res = {"text": text, "label": labels}
        return predict_res


class Option(object):
    def __init__(self, ):
        self.ro_pe = False      # rotated position encoding
        self.dropout = 0.1
        self.inner_dim = 64
        self.model_name = "GlobalPointer"
        self.best_model_path = ""
        self.ent2id = {"location": 0, "type": 1, "poiName": 2, "dishName": 3, "taste": 4}
        self.ner_type_num = len(self.ent2id)


def test():
    opt = Option()
    # set your trained models here
    opt.best_model_path = ""
    inf = Inference(opt)
    prompt = "请输入用户语句:"
    print(prompt)
    test_sentence = input()
    while test_sentence:
        labels = inf.evaluate(test_sentence)
        print(labels)
        prompt = "请输入用户语句:"
        print(prompt)
        test_sentence = input()


if __name__ == '__main__':
    test()
