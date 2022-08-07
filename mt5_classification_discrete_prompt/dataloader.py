import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer


def load_data(path, ind2label, prefix="下面是一则<extra_id_0>新闻？"):
    train = pd.read_csv(path, header=0, sep='\t', names=["text", "label"])
    print(train.shape)
    train = train.sample(10000, random_state=123)
    print("data shape:", train.shape)

    texts = train.text.apply(lambda x: prefix + x).to_list()
    labels = train.label.map(int).map(ind2label).apply(lambda x: "<extra_id_0>" + x).to_list()
    true_labels = train.label.map(int).to_list()
    print("data head", train.head())
    return texts, labels, true_labels


class TextDataset(Dataset):
    def __init__(self, filepath, ind2label):
        super(TextDataset, self).__init__()
        self.train, self.label, self.true_label = load_data(filepath, ind2label)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        label = self.label[item]
        true_label = self.true_label[item]
        return text, label, true_label


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=64, label_len=4):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_len = label_len

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        batch_true_label = [item[2] for item in batch]

        inputs = self.tokenizer(batch_text, max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(batch_label, max_length=self.label_len,
                                    truncation=True, padding='max_length', return_tensors='pt')

        return inputs, labels, batch_true_label


if __name__ == "__main__":
    from transformers import T5Tokenizer, MT5ForConditionalGeneration
    from model import MultiClassT5
    data_dir = "../data/THUCNews/news"
    pretrained_path = "/data/Learn_Project/Backup_Data/mt5-small"

    label_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8,
                  '财经': 9}
    ind2label_dict = dict(zip(list(label_dict.values()), list(label_dict.keys())))
    print(ind2label_dict)

    tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
    model_t5 = MT5ForConditionalGeneration.from_pretrained(pretrained_path)
    model = MultiClassT5(model_t5)

    text_dataset = TextDataset(os.path.join(data_dir, "test.txt"), ind2label_dict)
    text_dataset_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=text_dataset_call)

    # device = torch.device("cuda")
    for text, label, batch_true_label in text_dataloader:
        # print(text)
        # print(label)
        print(tokenizer.decode(text.input_ids[0]))
        print(tokenizer.decode(label.input_ids[0]))
        # text = text.to(device)
        out = model.generate(text)
        # print(out)
        predict = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(predict)
