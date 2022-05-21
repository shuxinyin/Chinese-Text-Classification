import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers

from dataloader import TextDataset, BatchTextCall, choose_bert_type
from model import MultiClass
from utils import load_config


def evaluation(model, test_dataloader, loss_func, label2ind_dict, save_path, valid_or_test="test"):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for ind, (token, segment, mask, label) in enumerate(test_dataloader):
        token = token.cuda()
        segment = segment.cuda()
        mask = mask.cuda()
        label = label.cuda()

        out = model(token, segment, mask)
        loss = loss_func(out, label)
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        predic = torch.max(out.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if valid_or_test == "test":
        report = metrics.classification_report(labels_all, predict_all, target_names=label2ind_dict.keys(), digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(test_dataloader), report, confusion
    return acc, total_loss / len(test_dataloader)


def train(config):
    label2ind_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True

    # load_data(os.path.join(data_dir, "cnews.train.txt"), label_dict)
    tokenizer, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)

    train_dataset = TextDataset(os.path.join(config.data_dir, "train.txt"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    valid_dataset = TextDataset(os.path.join(config.data_dir, "dev.txt"))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    test_dataset = TextDataset(os.path.join(config.data_dir, "test.txt"))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                 collate_fn=train_dataset_call)

    multi_classification_model = MultiClass(bert_encode_model, hidden_size=config.hidden_size,
                                            num_classes=10, pooling_type=config.pooling_type)
    multi_classification_model.cuda()
    # multi_classification_model.load_state_dict(torch.load(config.save_path))

    optimizer = torch.optim.AdamW(multi_classification_model.parameters(),
                                  lr=config.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.01, amsgrad=False)
    loss_func = F.cross_entropy

    if config.freeze_layer_count:
        # We freeze here the embeddings of the model
        for param in multi_classification_model.bert.embeddings.parameters():
            param.requires_grad = False

        if config.freeze_layer_count != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in multi_classification_model.bert.encoder.layer[:config.freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    loss_total, top_acc = [], 0
    for epoch in range(config.epoch):
        multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (token, segment, mask, label) in enumerate(tqdm_bar):
            token = token.cuda()
            segment = segment.cuda()
            mask = mask.cuda()
            label = label.cuda()

            multi_classification_model.zero_grad()
            out = multi_classification_model(token, segment, mask)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

        acc, loss, report, confusion = evaluation(multi_classification_model,
                                                  test_dataloader, loss_func, label2ind_dict,
                                                  config.save_path)
        print("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
        if top_acc < acc:
            top_acc = acc
            # torch.save(multi_classification_model.state_dict(), config.save_path)
            print(report, confusion)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert finetune test')
    parser.add_argument("--data_dir", type=str, default="../data/THUCNews/news_all")
    parser.add_argument("--save_path", type=str, default="../ckpt/bert_classification")
    parser.add_argument("--pretrained_path", type=str, default="/data/Learn_Project/Backup_Data/bert_chinese",
                        help="pre-train model path")
    parser.add_argument("--bert_type", type=str, default="bert", help="bert or albert")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="roberta-large:1024, bert/macbert/roberta-base:768, tiny_albert: 312")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--pooling_type", type=str, default="first-last-avg")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sent_max_len", type=int, default=44)
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Set this flag true if you are using an uncased model.")
    parser.add_argument("--bertadam", type=bool, default=False, help="If bertadam, then set correct_bias = False")
    parser.add_argument("--reinit_pooler", type=bool, default=True, help="reinit pooler layer")
    parser.add_argument("--reinit_layers", type=int, default=6, help="reinit pooler layers count")
    parser.add_argument("--frozen_layers", type=int, default=6, help="frozen layers")
    args = parser.parse_args()

    train(args)
