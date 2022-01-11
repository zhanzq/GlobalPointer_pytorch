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
import pickle as pkl

import torch

import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config_v2 import train_config as config
from transformers import BertConfig

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

    text = text.lower()
    return asr_correct("".join(REMAIN.findall(text)))


class XPNER(Dataset):
    def __init__(self, examples=None, vocab=21128):
        self.vocab = vocab
        if examples is None:
            examples = []
        self.examples = examples

    def add_examples(self, examples):
        self.examples.append(examples)

    def shuffle(self):
        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        for idx, val in enumerate(example["input_ids"]):
            if val >= self.vocab:
                example["input_ids"][idx] = 102
        self.examples[idx] = example
        input_ids_tensor = example["input_ids"] if type(example["input_ids"]) is torch.Tensor else torch.LongTensor(example["input_ids"])
        token_type_ids_tensor = example["token_type_ids"] if type(example["token_type_ids"]) is torch.Tensor else torch.LongTensor(example["token_type_ids"])
        attention_mask_tensor = example["attention_mask"] if type(example["attention_mask"]) is torch.Tensor else torch.LongTensor(example["attention_mask"])
        inputs = {
            "input_ids": input_ids_tensor,
            "token_type_ids": token_type_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }
        if "label" in self.examples[idx]:
            inputs["labels"] = example["label"] if type(example["label"]) is torch.Tensor else torch.LongTensor(example["label"])
        if "labels" in self.examples[idx]:
            inputs["labels"] = example["labels"] if type(example["labels"]) is torch.Tensor else torch.LongTensor(example["label"])
        if "start_pos" in self.examples[idx]:
            inputs["start_positions"] = example["start_pos"] if type(example["start_pos"]) is torch.Tensor else torch.LongTensor(example["label"])
        if "end_pos" in self.examples[idx]:
            inputs["end_positions"] = example["end_pos"] if type(example["end_pos"]) is torch.Tensor else torch.LongTensor(example["label"])
        if "weight" in self.examples[idx]:
            inputs["weights"] = example["weight"] if type(example["weight"]) is torch.Tensor else torch.LongTensor(example["weight"])
        return inputs


def load_ent_dct(data_path):
    assert os.path.exists(data_path), "entity dict file must exist"
    ent_dct = None
    with open(data_path, "r") as reader:
        line = reader.readline().strip()
        ent_dct = json.loads(line)

    return ent_dct


def load_ner_data_from_pkl(data_path):
    print("load data from {}".format(data_path))
    with open(data_path, "rb") as reader:
        data = pkl.load(reader)
        train_data = data["train"]
        valid_data = data["eval"]
        test_data = data["test"]

        print("train_data: {}, valid_data: {}, test_data: {}".format(len(train_data), len(valid_data), len(test_data)))
        return train_data, valid_data, test_data


def load_ner_data(data_dir, norm_text=True, tokenizer=None, opt=None, do_shuffle=True, splits=(0.7, 0.2, 0.1)):
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


def get_ner_dataset(lines, norm_text=True, tokenizer=None, opt=None):
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


def get_ner_example(sample, norm_text=True, tokenizer=None, opt=None):
    """

    :param sample: input data record, like {"text": "可以去吃无骨鱼啊", "label": [["dishName", "无骨鱼"]]}
    :param norm_text:
    :param tokenizer:
    :param opt: configure parameters
    :return:
    """

    if norm_text:
        sample["text"] = normalize(text=sample["text"])

    num_labels = opt.num_labels
    max_seq_len = opt.max_len
    label2id = opt.label2id
    text = sample["text"].lower()
    text = text[:max_seq_len-2]
    text_tokens = [ch for ch in text]

    inputs = tokenizer.encode_plus(text_tokens, max_length=max_seq_len, truncation=True,
                                   padding='max_length', add_special_tokens=True)

    anns = sample.get("label", [])
    entity_dct = {}
    for slot_type, slot_value in anns:
        slot_value = slot_value.lower()
        if slot_type in label2id and slot_value in text:
            if slot_type not in entity_dct:
                entity_dct[slot_type] = []
            entity_dct[slot_type].append(slot_value)
    labels = np.zeros((num_labels, max_seq_len, max_seq_len))

    entity_spans = get_entity_spans(text, entity_dct)
    for start, end, label in entity_spans:
        labels[label2id[label], start, end] = 1

    inputs["labels"] = labels
    data = {
        "sample": sample,
        "labels": torch.LongTensor(inputs["labels"]).to(opt.device),
        "input_ids": torch.LongTensor(inputs["input_ids"]).to(opt.device),
        "attention_mask": torch.LongTensor(inputs["attention_mask"]).to(opt.device),
        "token_type_ids": torch.LongTensor(inputs["token_type_ids"]).to(opt.device),
    }

    return data


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


def compute_ner_metrics(predictions):
    logits, labels = predictions
    pred = []
    true = []
    result = {i: [] for i in range(logits.shape[0])}
    for b, l, start, end in zip(*np.where(labels > 0)):
        true.append((b, l, start, end))
        result[b].append((l, start, end))
    for b, l, start, end in zip(*np.where(logits > 0)):
        pred.append((b, l, start, end))
        if (l, start, end) in result[b]:
            result[b].remove((l, start, end))
    is_correct = 0
    total = 0
    for idx in result:
        if not result[idx]:
            is_correct += 1
        total += 1
    acc = is_correct / total
    r = set(pred)
    t = set(true)
    x = len(r & t)
    y = len(r)
    z = len(t)
    try:
        f1, precision, recall = 2 * x / (y + z), x / y, x / z
    except ZeroDivisionError:
        f1, precision, recall = 0, 0, 0
    result = {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    return result


def main():
    data_path = "datasets/xp_ner_1124/train.jsonl"
    tokenizer = BertTokenizer.from_pretrained("/Users/zhanzq/Downloads/models/bert-base-chinese")
    with open(data_path, "r") as reader:
        lines = reader.readlines()
        dataset = get_ner_dataset(lines[:10], norm_text=True, tokenizer=tokenizer, opt=config)

        print("dataset samples:")
        for i in range(10):
            print("sample %d" % i)
            print(dataset[i])


def test():
    tokenizer = BertTokenizer.from_pretrained("/Users/zhanzq/Downloads/models/bert-base-chinese")
    sample = {"text": "可以去吃无骨鱼啊", "label": [["dishName", "无骨鱼"]]}
    data_i = get_ner_example(sample, norm_text=True, tokenizer=tokenizer, opt=config)
    print(data_i)


def load_data_from_pkl_test():
    data_path = "/Users/zhanzq/Downloads/mpcc-ner.pkl"
    train_data, valid_data, test_data = load_ner_data_from_pkl(data_path)


if __name__ == "__main__":
    config = BertConfig(**config)
    main()
    # test()
    # load_data_from_pkl_test()

