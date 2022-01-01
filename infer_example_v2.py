# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com
# date  : 2021/12/07
#

import torch
import numpy
from transformers import BertTokenizer
from models.gp_v2 import GlobalPointer
from data_utils_v2 import get_ner_example
from config_v2 import train_config


class Inference:
    """
    A simple inference example
    """
    def __init__(self, config):
        config.ro_pe = True
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)
        self.model = GlobalPointer.from_pretrained(config.pretrained_model_dir, config=config).to(config.device)

        print("loading pretrained model {0} from {1}".format(config.model_name, config.pretrained_model_dir))
        self.model.load_state_dict(torch.load(config.best_model_path, map_location=config.device))

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def get_ner(self, text, pred_matrix, threshold=-1.0):
        """
        pred_matrix: shape (1, ner_type_num, seq_len, seq_len)
        """
        id2label = self.config.id2label
        pred_matrix = pred_matrix.cpu().numpy()

        ent_lst = {}
        for _, ent_type_id, start_idx, end_idx in zip(*numpy.where(pred_matrix > threshold)):
            ner_type = id2label[ent_type_id]
            ent_text = text[start_idx-1: end_idx]
            entity = {"value": ent_text, "span": [start_idx-1, end_idx]}
            ent_type_lst = ent_lst.get(ner_type, [])
            ent_type_lst.append(entity)
            ent_lst[ner_type] = ent_type_lst

        return ent_lst

    def evaluate(self, text, threshold=0.0):
        sample = {"text": text}
        data = get_ner_example(sample, norm_text=True, tokenizer=self.tokenizer, opt=self.config)
        data["input_ids"] = data["input_ids"].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"].unsqueeze(0)
        data["token_type_ids"] = data["token_type_ids"].unsqueeze(0)
        pred_matrix = self.model(data["input_ids"], data["attention_mask"], data["token_type_ids"])[0]
        labels = self.get_ner(text, pred_matrix, threshold=threshold)
        predict_res = {"text": text, "label": labels}
        return predict_res


def interactive_test(config):
    # set your trained models here
    inf = Inference(config)
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


if __name__ == "__main__":
    train_config = GlobalPointer.config_class(**train_config)
    interactive_test(config=train_config)
