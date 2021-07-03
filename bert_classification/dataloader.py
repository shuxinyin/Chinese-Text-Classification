import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from bert_encode import choose_bert_type


def load_data(path, label_dic):
    train = pd.read_csv(path, header=None, sep='\t', names=["label", "text"])
    print(train.shape)
    # valid = pd.read_csv(os.path.join(path, "cnews.val.txt"), header=None, sep='\t', names=["label", "text"])
    # test = pd.read_csv(os.path.join(path, "cnews.test.txt"), header=None, sep='\t', names=["label", "text"])

    texts = train.text.to_list()
    labels = train.label.map(label_dic).to_list()
    # label_dic = dict(zip(train.label.unique(), range(len(train.label.unique()))))
    return texts, labels


class TextDataset(Dataset):
    def __init__(self, filepath, label_dict):
        super(TextDataset, self).__init__()
        self.train, self.label = load_data(filepath, label_dict)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        label = self.label[item]
        return text, label


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]

        batch_token, batch_segment, batch_mask = list(), list(), list()
        for text in batch_text:
            if len(text) > self.max_len-2:
                text = text[:self.max_len-2]
            token = self.tokenizer.tokenize(text)
            token = ['[CLS]'] + token + ['[SEP]']
            token_id = self.tokenizer.convert_tokens_to_ids(token)

            padding = [0] * (self.max_len - len(token_id))
            mask = [1] * len(token_id) + padding
            segment = [0] * len(token_id) + padding
            token_id = token_id + padding

            batch_token.append(token_id)
            batch_segment.append(segment)
            batch_mask.append(mask)

        batch_tensor_token = torch.tensor(batch_token)
        batch_tensor_segment = torch.tensor(batch_segment)
        batch_tensor_mask = torch.tensor(batch_mask)
        batch_tensor_label = torch.tensor(batch_label)
        return batch_tensor_token, batch_tensor_segment, batch_tensor_mask, batch_tensor_label


if __name__ == "__main__":

    data_dir = "../../data/THUCNews"
    pretrained_path = "D:\\Learn_Project\\Backup_Data\\tiny_bert_chinese_pretrained"

    label_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}

    # load_data(os.path.join(data_dir, "cnews.train.txt"), label_dict)
    tokenizer, model = choose_bert_type(pretrained_path, bert_type="tiny_albert")

    text_dataset = TextDataset(os.path.join(data_dir, "cnews.train.txt"), label_dict)
    text_dataset_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=text_dataset_call)

    for i, (token, segment, mask, label) in enumerate(text_dataloader):
        print(i, token, segment, mask, label)
        break
