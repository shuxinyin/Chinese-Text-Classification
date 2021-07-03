import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer


def bert_encode(batch_token, batch_segment, batch_attention_mask, model):
    """after complete the selection of bert type, go on bert encode
    return: cls vector
    """
    # tensor_token = torch.tensor(np.random.randint(low=103, high=1000, size=[2, 16], dtype="int64"))
    # text_encode = torch.tensor(tokenizer.encode(batch_text)).unsqueeze(0)
    print(batch_token.shape)
    with torch.no_grad():
        outputs = model(batch_token, token_type_ids=batch_segment, attention_mask=batch_attention_mask)
        outputs = outputs[0][:, 0, :]
        # fc
        # cross_entropy

    return outputs


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


class BatchText(Dataset):
    def __init__(self, total_text):
        super(BatchText, self).__init__()
        self.text = total_text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_text):
        max_len = max(len(t) + 2 for t in batch_text)

        batch_token, batch_segment, batch_mask = list(), list(), list()
        for text in batch_text:
            if len(text) > 510:
                text = text[:510]
            token = tokenizer.tokenize(text)
            token = ['[CLS]'] + token + ['[SEP]']
            token_id = tokenizer.convert_tokens_to_ids(token)

            padding = [0] * (max_len - len(token_id))
            mask = [1] * len(token_id) + padding
            segment = [0] * len(token_id) + padding
            token_id = token_id + padding

            batch_token.append(token_id)
            batch_segment.append(segment)
            batch_mask.append(mask)

        batch_tensor_token = torch.tensor(batch_token)
        batch_tensor_segment = torch.tensor(batch_segment)
        batch_tensor_mask = torch.tensor(batch_mask)

        return batch_tensor_token, batch_tensor_segment, batch_tensor_mask



if __name__ == "__main__":
    path = "D:\\Learn_Project\\Backup_Data\\tiny_bert_chinese_pretrained"
    # path = "D:\\Learn_Project\\Backup_Data\\macbert_chinese_pretrained"

    text = ["清晨去学校，小鸟对我笑。", "小鸟说，早早早， 是谁一早背上大书包。"] * 1000

    tokenizer, model = choose_bert_type(path, bert_type="tiny_albert")

    text_dataset = BatchText(text)
    text_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=4, shuffle=True, collate_fn=text_call)

    tqdm_bar = tqdm(text_dataloader, desc="Training")
    for step, (batch_token, batch_segment, batch_attention_mask) in enumerate(tqdm_bar):
        # print(batch_token.shape)
        output = bert_encode(batch_token, batch_segment, batch_attention_mask, model)
        print("--", output[0][:, 0, :].shape)
        sys.exit()


