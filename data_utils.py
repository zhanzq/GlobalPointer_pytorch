# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/16
#

import os
import re
import json
import random
import traceback

import torch

import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


REMAIN = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9.]+")
ASR_RE = [
    (
        re.compile(r"(小批|小屁|小平|小佩|小片|小弟|小D|小贝)"),
        "小P"
    ),
    (
        re.compile(r"(牌号|拍号)"),
        "排号"
    ),
    (
        re.compile(r"拍个号"),
        "排个号"
    )
]


def asr_correct(text):
    for compiler, sub in ASR_RE:
        text = compiler.sub(sub, text)
    return text


def normalize(text):
    """
    1. 去掉无关字符
    2. ASR纠错通过词表替换

    :param text:
    :return:
    """

    return asr_correct("".join(REMAIN.findall(text)))


class Option(object):
    def __init__(self, max_seq_len=50, ner_type_num=5):
        self.max_seq_len = max_seq_len
        self.ner_type_num = ner_type_num
        self.ent2id = {"location": 0, "type": 1, "poiName": 2, "dishName": 3, "taste": 4}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_entity_spans(text, entity_dct):
    entity_spans = []
    for slot_type in entity_dct:
        entity_lst = entity_dct[slot_type]
        for entity in entity_lst:
            entity_len = len(entity)
            for idx in range(len(text)):
                if text[idx: idx + entity_len] == entity:
                    entity_spans.append([idx + 1, idx + entity_len, slot_type])
                    # print("span: [%d, %d]" % (idx + 1, idx + entity_len))
    return entity_spans


def get_ner_example(sample, norm_text=False, tokenizer=None, opt=None):
    """

    :param sample: input data record, like {"text": "可以去吃无骨鱼啊", "label": [["dishName", "无骨鱼"]]}
    :param norm_text:
    :param tokenizer:
    :param opt: configure parameters
    :return:
    """

    if norm_text:
        sample["text"] = normalize(text=sample["text"])

    ner_type_num = opt.ner_type_num
    max_seq_len = opt.max_seq_len
    ent2id = opt.ent2id
    text = sample["text"].lower()
    text_tokens = [ch for ch in text]

    inputs = tokenizer.encode_plus(text_tokens, max_length=max_seq_len, truncation=True,
                                   padding='max_length', add_special_tokens=True)

    anns = sample.get("label", None)
    entity_dct = {}
    for slot_type, slot_value in anns:
        slot_value = slot_value.lower()
        if slot_type in ent2id and slot_value in text:
            if slot_type not in entity_dct:
                entity_dct[slot_type] = []
            entity_dct[slot_type].append(slot_value)
    labels = np.zeros((ner_type_num, max_seq_len, max_seq_len))

    entity_spans = get_entity_spans(text, entity_dct)
    for start, end, label in entity_spans:
        labels[ent2id[label], start, end] = 1

    inputs["labels"] = labels
    data = {
        "sample": sample,
        "labels": torch.LongTensor(inputs["labels"]).to(opt.device),
        "input_ids": torch.LongTensor(inputs["input_ids"]).to(opt.device),
        "attention_mask": torch.LongTensor(inputs["attention_mask"]).to(opt.device),
        "token_type_ids": torch.LongTensor(inputs["token_type_ids"]).to(opt.device),
    }

    return data


def get_ner_dataset(lines, norm_text=False, tokenizer=None, opt=None):
    all_data = []
    for line in lines:
        line = line.strip()
        try:
            sample = json.loads(line)
            data = get_ner_example(sample, norm_text=norm_text, tokenizer=tokenizer, opt=opt)
            all_data.append(data)
        except Exception as e:
            print(e)
            traceback.print_exc()

    return all_data


def load_ner_data(data_dir, norm_text=False, tokenizer=None, opt=None, do_shuffle=True, splits=(0.7, 0.2, 0.1)):
    train_lines, valid_lines, test_lines = [], [], []
    assert os.path.isdir(data_dir), "input path must be data dir"
    files = os.listdir(data_dir)
    for file_name in files:
        data_path = os.path.join(data_dir, file_name)
        if "train.jsonl" == file_name:
            with open(data_path, "r", encoding="utf-8", newline="\n", errors="ignore") as reader:
                lines = reader.readlines()
                train_lines.extend(lines)
        elif "dev.jsonl" == file_name:
            with open(data_path, "r", encoding="utf-8", newline="\n", errors="ignore") as reader:
                lines = reader.readlines()
                valid_lines.extend(lines)
        elif "test.jsonl" == file_name:
            with open(data_path, "r", encoding="utf-8", newline="\n", errors="ignore") as reader:
                lines = reader.readlines()
                test_lines.extend(lines)
        else:
            print("irrelevant data file: {:s}".format(data_path))
    train_sz = len(train_lines)
    if do_shuffle:
        random.shuffle(train_lines)
    total_sz = train_sz
    if len(test_lines) == 0:
        test_sz = int(total_sz*splits[2])
        train_sz -= test_sz
        test_lines = train_lines[-test_sz:]
    if len(valid_lines) == 0:
        valid_sz = int(total_sz*splits[1])
        train_sz -= valid_sz
        valid_lines = train_lines[train_sz:train_sz + valid_sz]

    train_data = get_ner_dataset(train_lines, norm_text=norm_text, tokenizer=tokenizer, opt=opt)
    valid_data = get_ner_dataset(valid_lines, norm_text=norm_text, tokenizer=tokenizer, opt=opt)
    test_data = get_ner_dataset(test_lines, norm_text=norm_text, tokenizer=tokenizer, opt=opt)

    return train_data, valid_data, test_data


class XPNER(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


def main():
    data_path = "datasets/xp_ner_1124/train.jsonl"
    opt = Option(max_seq_len=20)
    tokenizer = BertTokenizer.from_pretrained("/Users/zhanzq/Downloads/models/bert-base-chinese")
    with open(data_path, "r") as reader:
        lines = reader.readlines()
        dataset = get_ner_dataset(lines[:10], norm_text=False, tokenizer=tokenizer, opt=opt)

        print("dataset samples:")
        for i in range(10):
            print("sample %d" % i)
            print(dataset[i])


def test():
    opt = Option(max_seq_len=10)
    tokenizer = BertTokenizer.from_pretrained("/Users/zhanzq/Downloads/models/bert-base-chinese")
    sample = {"text": "可以去吃无骨鱼啊", "label": [["dishName", "无骨鱼"]]}
    data_i = get_ner_example(sample, norm_text=False, tokenizer=tokenizer, opt=opt)
    print(data_i)


if __name__ == "__main__":
    # main()
    test()
