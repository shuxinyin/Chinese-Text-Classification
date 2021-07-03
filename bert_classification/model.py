import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClass(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    """

    def __init__(self, bert_encode_model, hidden_size=768, num_classes=10):
        super(MultiClass, self).__init__()
        self.bert_encode_model = bert_encode_model
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch_token, batch_segment, batch_attention_mask):
        with torch.no_grad():
            outputs = self.bert_encode_model(batch_token, token_type_ids=batch_segment,
                                             attention_mask=batch_attention_mask)
            outputs = outputs[0][:, 0, :]
        out = self.fc(outputs)
        # labels = torch.nn.functional.one_hot(labels, self.num_classes)

        return out
