# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/16
#

import os
import sys
import math

import numpy
import random
import logging
import argparse

from transformers import BertModel, BertTokenizer
from time import strftime, localtime

import torch
from torch.utils.data import DataLoader

from data_utils import XPNER
from data_utils import load_ner_data

from models.gp import GlobalPointer

from common.utils import multilabel_categorical_crossentropy


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        data_dir = opt.data_dir
        train_data, valid_data, test_data = load_ner_data(
            data_dir=data_dir,
            norm_text=False,
            tokenizer=self.tokenizer,
            do_shuffle=True,
            opt=opt,
        )

        self.train_dataset = XPNER(train_data)
        self.valid_dataset = XPNER(valid_data)
        self.test_dataset = XPNER(test_data)

        if opt.device.type == "cuda":
            logger.info("cuda memory allocated: {}".format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_untrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_untrainable_params += n_params
        logger.info("> n_trainable_params: %d, n_untrainable_params: %d" % (n_trainable_params, n_untrainable_params))
        logger.info("> training arguments:")
        for arg in vars(self.opt):
            logger.info(">>> {0}: {1}".format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdev = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdev, b=stdev)

    @staticmethod
    def loss_fun(gd_truths, preds):
        """
        gd_truths: (batch_size, ner_type_num, seq_len, seq_len)
        preds: (batch_size, ner_type_num, seq_len, seq_len)
        """
        batch_size, ner_type_num = preds.shape[:2]
        gd_truths = gd_truths.reshape(batch_size*ner_type_num, -1)
        preds = preds.reshape(batch_size*ner_type_num, -1)
        loss = multilabel_categorical_crossentropy(gd_truths, preds)
        return loss

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, model_name="None", tag=0):
        path = None
        max_val_f1 = 0
        max_val_acc = 0
        global_step = 0
        max_val_epoch = 0
        for i_epoch in range(self.opt.num_epoch):
            logger.info(">" * 100)
            logger.info("epoch: {}".format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                outputs = self.model(batch)
                targets = batch["labels"].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_sz = len(outputs)
                batch_f1, batch_acc = self.get_sample_fp(outputs, targets)
                n_total += batch_sz
                batch_loss = loss.item()
                loss_total += batch_loss * batch_sz
                if global_step % self.opt.log_step == 0:
                    # train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info("steps: %d, total avg loss: %.4f, batch_loss: %.4f, batch_f1: %.4f, \
batch_acc: %.4f" % (global_step, train_loss, batch_loss, batch_f1, batch_acc))

            val_res = self._evaluate_acc_f1(val_data_loader)
            val_acc, val_f1 = val_res["P"], val_res["F1"]
            logger.info("> val_acc: {:.4f}, val_f1: {:.4f}".format(val_acc, val_f1))
            if val_acc > max_val_acc:
                if not os.path.exists("state_dict"):
                    os.mkdir("state_dict")
                path = "state_dict/{0}_{1}_val_acc_{2}_{3}_{4}".format(
                    self.opt.model_name, self.opt.dataset, round(val_acc, 4), model_name, tag)
                torch.save(self.model.state_dict(), path)
                logger.info(">> saved better model: {}".format(path))
                org_path = "state_dict/{0}_{1}_val_acc_{2}_{3}_{4}".format(
                    self.opt.model_name, self.opt.dataset, round(max_val_acc, 4), model_name, tag)
                if os.path.exists(org_path):
                    os.remove(org_path)
                    logger.info(">> remove older model: {}".format(org_path))

                max_val_acc = val_acc
                max_val_epoch = i_epoch
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print(">> early stop.")
                break

        return path

    @staticmethod
    def get_sample_fp(y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        f1 = 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred + 1e-5)
        p = torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)
        return f1, p

    def _evaluate_acc_f1(self, data_loader):
        preds, gd_truths = [], []
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_targets = t_batch["labels"].to(self.opt.device)
                t_outputs = self.model(t_batch)

                batch_truths = t_targets.cpu().numpy()
                batch_preds = t_outputs.cpu().numpy()
                for b, l, start, end in zip(*numpy.where(batch_preds > 0)):
                    preds.append((i_batch, b, l, start, end))
                for b, l, start, end in zip(*numpy.where(batch_truths > 0)):
                    gd_truths.append((i_batch, b, l, start, end))

        R = set(preds)
        T = set(gd_truths)
        X = len(R & T) + 1e-5
        Y = len(R) + 1e-5
        Z = len(T) + 1e-5
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        result = {
            "P": precision,
            "R": recall,
            "F1": f1,
        }
        return result

    def run(self, model_name="chinese-bert-base", tag=0):
        # Loss and Optimizer
        criterion = self.loss_fun
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, model_name, tag)
        self.model.load_state_dict(torch.load(best_model_path))
        print("load model from %s" % best_model_path)
        train_res = self._evaluate_acc_f1(train_data_loader)
        val_res = self._evaluate_acc_f1(val_data_loader)
        test_res = self._evaluate_acc_f1(test_data_loader)
        logger.info(">> train_p: {:.4f}, train_r: {:.4f}, train_f1: {:.4f}".format(train_res["P"],
                                                                                   train_res["R"], train_res["F1"]))
        logger.info(">> val_p: {:.4f}, val_r: {:.4f}, val_f1: {:.4f}".format(val_res["P"],
                                                                             val_res["R"], val_res["F1"]))
        logger.info(">> test_p: {:.4f}, test_r: {:.4f}, test_f1: {:.4f}".format(test_res["P"],
                                                                                test_res["R"], test_res["F1"]))

        return train_res, val_res, test_res


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gp", type=str)
    parser.add_argument("--dataset", default="xp_ner", type=str, help="xp_ner dataset")
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--initializer", default="xavier_uniform_", type=str)
    parser.add_argument("--lr", default=2e-5, type=float, help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument("--num_epoch", default=5, type=int, help="try larger number (>20) for non-BERT models")
    parser.add_argument("--batch_size", default=32, type=int, help="try 16, 32, 64 for BERT models")
    parser.add_argument("--log_step", default=10, type=int)
    parser.add_argument("--embed_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=300, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--inner_dim", default=64, type=int)
    parser.add_argument("--bert_dim", default=768, type=int)
    parser.add_argument("--pretrained_bert_name", default="bert-base-uncased", type=str)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--ner_type_num", default=6, type=int)
    parser.add_argument("--hops", default=3, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--ro_pe", default=False, type=bool)
    parser.add_argument("--device", default=None, type=str, help="e.g. cuda:0")
    parser.add_argument("--data_dir", default="datasets/xp_ner_1129/", type=str, help="xp dataset")
    parser.add_argument("--seed", default=1234, type=int, help="set seed for reproducibility")
    parser.add_argument("--valid_dataset_ratio", default=0.1, type=float,
                        help="set ratio between 0 and 1 for validation support")

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(opt.seed)

    model_classes = {
        'gp': GlobalPointer,
    }
    dataset_files = {
        'xp_ner': {
            'train': './datasets/xp_ner/train.jsonl',
            'dev': './datasets/xp_ner/dev.jsonl',
            'test': './datasets/xp_ner/test.jsonl',
        },
    }

    input_cols = {
        'gp': ['input_ids', 'attention_mask', 'token_type_ids', "labels"],
    }

    initializers = {
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "orthogonal_": torch.nn.init.orthogonal_,
    }
    optimizers = {
        "sgd": torch.optim.SGD,
        "asgd": torch.optim.ASGD,  # default lr=0.01
        "adam": torch.optim.Adam,  # default lr=0.001
        "adamax": torch.optim.Adamax,  # default lr=0.002
        "rmsprop": torch.optim.RMSprop,  # default lr=0.01
        "adagrad": torch.optim.Adagrad,  # default lr=0.01
        "adadelta": torch.optim.Adadelta,  # default lr=1.0
    }
    opt.optimizer = optimizers[opt.optimizer]
    opt.inputs_cols = input_cols[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.initializer = initializers[str(opt.initializer)]
    opt.ent2id = {"location": 0, "type": 1, "poiName": 2, "dishName": 3, "taste": 4, "listNum"}
    opt.ner_type_num = len(opt.ent2id)
    if opt.device is None:
        if torch.cuda.is_available():
            opt.device = torch.device("cuda")
        else:
            opt.device = torch.device("cpu")
    else:
        opt.device = torch.device(str(opt.device))

    log_file = "{}-{}-{}.log".format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    results = []
    run_num = 5
    model_name = opt.pretrained_bert_name.split("/")[-1]
    with open("train.log", "a") as writer:
        for i in range(run_num):
            ins = Instructor(opt)
            res_i = ins.run(model_name=model_name, tag=i)
            result_i = []
            for res_ij in res_i:
                result_i.extend([res_ij["P"], res_ij["F1"]])
            results.append(result_i)
        avg_result = numpy.mean(results, axis=0)
        avg_result = [it.item() for it in avg_result]
        # write logs
        writer.write("pretrained model: {:s}, model architecture: {:s}\n".format(model_name, opt.model_name))
        writer.write("{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}\n".format("run_idx", "train_acc", "train_f1",
                                                                           "valid_acc", "valid_f1",
                                                                           "test_acc", "test_f1"))
        for i in range(run_num):
            writer.write("%-10d%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f\n" % tuple([i] + results[i]))
        writer.write("%-10s%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f\n\n\n" % tuple(["avg"] + avg_result))


if __name__ == "__main__":
    main()
