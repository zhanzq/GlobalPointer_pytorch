# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/16
#

import torch
import torch.nn as nn


class GlobalPointer(nn.Module):
    def __init__(self, bert, opt):
        super(GlobalPointer, self).__init__()
        self.bert = bert
        opt.hidden_size = bert.config.hidden_size
        self.ro_pe = opt.ro_pe
        self.inner_dim = opt.inner_dim
        self.ner_type_num = opt.ner_type_num
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.hidden_size, opt.ner_type_num * opt.inner_dim * 2)

    @staticmethod
    def sinusoidal_position_embedding(batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        # embeddings = embeddings.to(self.device)
        return embeddings

    def rotate_position_embedding(self, qw, kw):
        # qw shape: (batch_size, seq_len, ner_type_num, inner_dim)
        batch_size = qw.size()[0]
        seq_len = qw.size()[1]
        inn_dim = qw.size()[3]

        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, inn_dim)

        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, 0::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., 0::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., 0::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def get_logits(self, bert_outputs, attention_mask):
        # shape (batch_size, seq_len, hidden_size)
        last_hidden_state = bert_outputs[0]
        outputs = self.dropout(last_hidden_state)

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(outputs)  # shape (batch_size, seq_len, ner_type_num*inner_dim*2)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)  # shape (batch_size, seq_len, ner_type_num, inner_dim*2)

        # qw,kw:(batch_size, seq_len, ner_type_num, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.ro_pe:
            qw, kw = self.rotate_position_embedding(qw, kw)

        # logits:(batch_size, ner_type_num, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ner_type_num, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        logits = logits / self.inner_dim ** 0.5

        return logits

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)

        logits = self.get_logits(bert_outputs=bert_outputs, attention_mask=attention_mask)

        return logits
