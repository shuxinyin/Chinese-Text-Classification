import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):

    def __init__(self, params):
        super(CNNClassifier, self).__init__()

        self.num_kernels = params.num_kernels
        self.kernel_size = params.kernel_size
        self.num_class = params.num_class
        self.stride = params.stride

        if params.pretrained_weight:
            word_model_emb = np.load(params.pretrained_weight_path)
            weights = torch.tensor(word_model_emb)
            self.emb = nn.Embedding.from_pretrained(weights)
        else:
            self.emb = nn.Embedding(params.vocab_size, params.embedding_dim, padding_idx=params.padding_index)

        self.conv_0 = nn.Conv2d(1, self.num_kernels, (self.kernel_size[0], params.embedding_dim), self.stride)
        self.conv_1 = nn.Conv2d(1, self.num_kernels, (self.kernel_size[1], params.embedding_dim), self.stride)
        self.conv_2 = nn.Conv2d(1, self.num_kernels, (self.kernel_size[2], params.embedding_dim), self.stride)

        # conv output size: [(w-k)+2p]/s+1
        # (batch, channel=1, seq_len, emb_size)
        self.fc = nn.Linear(len(self.kernel_size) * self.num_kernels, self.num_class)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, text):
        # input: (B, 512, 312)
        # text (batch, channel=1, seq_len)

        emb = self.emb(text)  # (batch, seq_len, emb_size)
        emb = emb.unsqueeze(dim=1)  # (batch, channel=1, seq_len, emb_size)

        # after conv: (batch, num_kernels, seq_len - kernel_size[0] + 1, 1)
        conved0 = F.relu(self.conv_0(emb).squeeze(3))
        conved1 = F.relu(self.conv_1(emb).squeeze(3))
        conved2 = F.relu(self.conv_2(emb).squeeze(3))


        # pooled: (batch, n_channel)
        pool0 = nn.MaxPool1d(conved0.shape[2], self.stride)
        pool1 = nn.MaxPool1d(conved1.shape[2], self.stride)
        pool2 = nn.MaxPool1d(conved2.shape[2], self.stride)

        pooled0 = pool0(conved0).squeeze(2)
        pooled1 = pool1(conved1).squeeze(2)
        pooled2 = pool2(conved2).squeeze(2)

        # (batch, n_chanel * num_filters)
        cat_pool = torch.cat([pooled0, pooled1, pooled2], dim=1)
        fc = self.fc(cat_pool)
        return fc


class Param(object):
    def __init__(self, kernel_size=[3, 4, 5], num_kernels=100, vocab_size=100, embedding_dim=64, num_class=10, stride=1,
                 padding_index=0, dropout=0.3):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        if kernel_size is None:
            kernel_size = [3, 4, 5]
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        self.padding_index = padding_index
        self.stride = stride

        self.num_class = num_class

        self.dropout = dropout


if __name__ == '__main__':
    import numpy as np

    params = Param()
    print(params.kernel_size[0])

    model = CNNClassifier(params)

    text = np.random.randint(low=0, high=100, size=[8, 20])
    text = torch.tensor(text).to(torch.int64)
    print(text.shape)
    fc_res = model(text)
