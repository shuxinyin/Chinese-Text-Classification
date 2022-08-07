import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

import torch
from torch import nn
from transformers import MT5Model, T5Tokenizer, MT5ForConditionalGeneration

from torch.nn import CrossEntropyLoss


class MultiClassT5(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    refer from t5 sentiment task: https://github.com/huggingface/transformers/issues/3704
    """

    def __init__(self, mt5_model, pooling_type='first-last-avg'):
        super(MultiClassT5, self).__init__()
        self.mt5 = mt5_model
        self.pooling = pooling_type

    def forward(self, inputs, labels):
        out = self.mt5(**inputs, labels=labels["input_ids"])
        loss = out.loss

        return loss

    def generate(self, inputs):
        out = self.mt5.generate(inputs.input_ids)
        # predicted_tokens = self.mt5.generate(inputs.input_ids, decoder_start_token_id=tokenizer.pad_token_id,
        #                                      num_beams=5, early_stopping=True, max_length=4)
        return out


def test_batch_train():
    import os
    from dataloader import BatchTextCall, TextDataset
    from torch.utils.data import DataLoader
    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    model_t5 = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)

    label2ind_dict = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5,
                      'politics': 6, 'sports': 7, 'game': 8, 'entertainment': 9}
    ind2label_dict = dict(zip(list(label2ind_dict.values()), list(label2ind_dict.keys())))

    multi_classification_model = MultiClassT5(model_t5)
    train_dataset_call = BatchTextCall(tokenizer, max_len=64)

    train_dataset = TextDataset(os.path.join("../data/THUCNews/news", "train.txt"), ind2label_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6,
                                  collate_fn=train_dataset_call)

    for i, (inputs, labels, true_labels) in enumerate(train_dataloader):
        # print(inputs["input_ids"].shape, labels["input_ids"].shape)

        outputs = multi_classification_model(inputs, labels)
        print(outputs)
        sys.exit()


if __name__ == '__main__':
    test_batch_train()
