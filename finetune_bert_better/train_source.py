# code reference: https://github.com/asappresearch/revisit-bert-finetuning

import os
import sys
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer

from dataloader import TextDataset, BatchTextCall
from model import MultiClass
from utils import load_config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d:%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert（bert）
    return: tokenizer, model
    """

    if bert_type == "albert":
        model_config = BertConfig.from_pretrained(path)
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "bert" or bert_type == "roberta":
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    else:
        model_config, model = None, None
        print("ERROR, not choose model!")

    return model_config, model


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

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)

    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)

    train_dataset = TextDataset(os.path.join(config.data_dir, "train.txt"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    valid_dataset = TextDataset(os.path.join(config.data_dir, "dev.txt"))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    test_dataset = TextDataset(os.path.join(config.data_dir, "test.txt"))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=10,
                                 collate_fn=train_dataset_call)

    model_config, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    multi_classification_model = MultiClass(bert_encode_model, model_config,
                                            num_classes=10, pooling_type=config.pooling_type)
    multi_classification_model.cuda()
    # multi_classification_model.load_state_dict(torch.load(config.save_path))

    if config.weight_decay:
        param_optimizer = list(multi_classification_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=config.lr, correct_bias=not config.bertadam)

    else:
        optimizer = transformers.AdamW(multi_classification_model.parameters(), lr=config.lr, betas=(0.9, 0.999),
                                       eps=1e-08, weight_decay=0.01, correct_bias=not config.bertadam)

    num_train_optimization_steps = len(train_dataloader) * config.epoch
    if config.warmup_proportion != 0:
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 int(num_train_optimization_steps * config.warmup_proportion),
                                                                 num_train_optimization_steps)
    else:
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 int(num_train_optimization_steps * config.warmup_proportion),
                                                                 num_train_optimization_steps)

    loss_func = F.cross_entropy

    # reinit pooler-layer
    if config.reinit_pooler:
        if config.bert_type in ["bert", "roberta", "albert"]:
            logger.info(f"reinit pooler layer of {config.bert_type}")
            encoder_temp = getattr(multi_classification_model, config.bert_type)
            encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
        else:
            raise NotImplementedError

    # reinit encoder layers
    if config.reinit_layers > 0:
        if config.bert_type in ["bert", "roberta", "albert"]:
            # assert config.reinit_pooler
            logger.info(f"reinit  layers count of {str(config.reinit_layers)}")

            encoder_temp = getattr(multi_classification_model, config.bert_type)
            for layer in encoder_temp.encoder.layer[-config.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
        else:
            raise NotImplementedError

    if config.freeze_layer_count:
        logger.info(f"frozen layers count of {str(config.freeze_layer_count)}")
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
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
        logger.info("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

        acc, loss, report, confusion = evaluation(multi_classification_model,
                                                  test_dataloader, loss_func, label2ind_dict,
                                                  config.save_path)
        logger.info("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
        if top_acc < acc:
            top_acc = acc
            # torch.save(multi_classification_model.state_dict(), config.save_path)
            logger.info(f"{report} \n {confusion}")
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert finetune test')
    parser.add_argument("--data_dir", type=str, default="../data/THUCNews/news")
    parser.add_argument("--save_path", type=str, default="../ckpt/bert_classification")
    parser.add_argument("--pretrained_path", type=str, default="/data/Learn_Project/Backup_Data/bert_chinese",
                        help="pre-train model path")
    parser.add_argument("--bert_type", type=str, default="bert", help="bert or albert")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--pooling_type", type=str, default="first-last-avg")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sent_max_len", type=int, default=44)
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Set this flag true if you are using an uncased model.")
    parser.add_argument("--bertadam", type=int, default=0, help="If bertadam, then set correct_bias = False")
    parser.add_argument("--weight_decay", type=int, default=0, help="If weight_decay, set 1")
    parser.add_argument("--reinit_pooler", type=int, default=1, help="reinit pooler layer")
    parser.add_argument("--reinit_layers", type=int, default=6, help="reinit pooler layers count")
    parser.add_argument("--freeze_layer_count", type=int, default=6, help="freeze layers count")
    args = parser.parse_args()

    log_filename = f"test_bertadam{args.bertadam}_weight_decay{str(args.weight_decay)}" \
                   f"_reinit_pooler{str(args.reinit_pooler)}_reinit_layers{str(args.reinit_layers)}" \
                   f"_frozen_layers{str(args.freeze_layer_count)}_warmup_proportion{str(args.warmup_proportion)}"
    logger.addHandler(logging.FileHandler(os.path.join("./log", log_filename), 'w'))
    logger.info(args)

    train(args)
