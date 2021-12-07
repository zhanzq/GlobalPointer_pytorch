# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/12/07
#

import torch
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import functional


def multilabel_ce_loss(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def compute_kl_loss(logits_1, logits_2, pad_mask=None):
    p_loss = functional.kl_div(
        functional.log_softmax(logits_1, dim=-1),
        functional.softmax(logits_2, dim=-1),
        reduction='none'
    )
    q_loss = functional.kl_div(
        functional.log_softmax(logits_2, dim=-1),
        functional.softmax(logits_1, dim=-1),
        reduction='none'
    )

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class GlobalPointer(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dense = torch.nn.Linear(config.hidden_size, 2 * config.num_labels * config.inner_dim)
        self.init_weights()

    def get_logits(self, outputs, attention_mask):
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size = sequence_output.size()[0]
        seq_len = sequence_output.size()[1]
        proj_outputs = self.dense(sequence_output)
        proj_outputs = torch.split(proj_outputs, self.config.inner_dim * 2, dim=-1)
        proj_outputs = torch.stack(proj_outputs, dim=-2)
        qw, kw = proj_outputs[..., : self.config.inner_dim], proj_outputs[..., self.config.inner_dim:]
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(
            batch_size, self.config.num_labels, seq_len, seq_len)
        pad_mask = pad_mask.to(logits.dtype)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        logits = logits / self.config.inner_dim ** 0.5
        return logits

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        config = getattr(self, "config")

        return_dict = return_dict if return_dict is not None else config.use_return_dict
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.get_logits(outputs, attention_mask)

        loss = None
        if labels is not None:
            batch_size = outputs[0].size()[0]
            loss = multilabel_ce_loss(
                logits.reshape(batch_size * config.num_labels, -1),
                labels.reshape(batch_size * config.num_labels, -1)
            )

            if config.use_r_drop is True:
                outputs_reg = self.base_model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                logits_reg = self.get_logits(outputs_reg, attention_mask)
                loss += 0.05 * compute_kl_loss(logits_reg, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
