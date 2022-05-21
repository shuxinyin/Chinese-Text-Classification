import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


def load_data(path, label_dic):
    train = pd.read_csv(path, header=0, sep='\t', names=["text", "label"])
    print(train.shape)

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
            if len(text) > self.max_len - 2:
                text = text[:self.max_len - 2]
            token = self.tokenizer.tokenize(text)
            token = ['[CLS]'] + token + ['[SEP]']
            token_id = self.tokenizer.convert_tokens_to_ids(token)

            padding = [0] * (self.max_len - len(token_id))
            token_id = token_id + padding
            batch_token.append(token_id)

        batch_tensor_token = torch.tensor(batch_token)
        batch_tensor_label = torch.tensor(batch_label)
        return batch_tensor_token, batch_tensor_label


from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer


def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert
    return: tokenizer, model
    """
    tokenizer = BertTokenizer.from_pretrained(path)
    model_config = BertConfig.from_pretrained(path)
    if bert_type == "tiny_albert":
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "macbert":
        model = BertModel.from_pretrained(path, config=model_config)
    else:
        model = None
        print("ERROR, not choose model!")
    return tokenizer, model


if __name__ == "__main__":

    data_dir = "../../news_all/THUCNews"
    pretrained_path = "D:\\Learn_Project\\Backup_Data\\albert_chinese_pretrained"

    label_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}

    # load_data(os.path.join(data_dir, "cnews.train.txt"), label_dict)
    tokenizer, model = choose_bert_type(pretrained_path, bert_type="tiny_albert")

    text_dataset = TextDataset(os.path.join(data_dir, "cnews.train.txt"), label_dict)
    text_dataset_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=text_dataset_call)

    for i, (token, label) in enumerate(text_dataloader):
        print(i, token, label)
        break
