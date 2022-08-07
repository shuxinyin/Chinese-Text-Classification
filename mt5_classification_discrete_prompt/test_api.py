import re
import torch
import numpy as np
from sklearn import metrics

from transformers import MT5Model, T5Tokenizer, MT5ForConditionalGeneration

def test_report():
    labels_all = np.array(['apples', 'foobar', 'cowboy'])
    predict_all = np.array(['apples', 'foobar', 'cowboy'])
    acc = metrics.accuracy_score(labels_all, predict_all)

    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print(acc)
    print(report)
    print(confusion)


def filter_special_token(s):
    s = re.sub('<pad>|</s>|\s+', "", s)
    return s

def t5_demo():
    device = torch.device("cuda")

    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    model = MT5Model.from_pretrained(mt5_pretrain)
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)
    article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    summary = "Weiter Verhandlung in Syrien."
    inputs = tokenizer(article, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summary, return_tensors="pt")

    print(inputs)
    print(labels)
    param_optimizer = list(model.named_parameters())
    print("---", param_optimizer)

    outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    hidden_states = outputs.last_hidden_state
    print(hidden_states.shape)


def t5_demo_generate():
    ''' 文本生成
    '''
    from transformers import MT5ForConditionalGeneration, T5Tokenizer

    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    # model = MultiClassT5("mt5")
    model = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)
    tokens = ['en', 'fr', 'zh']
    tokenizer.add_special_tokens({'additional_special_tokens': tokens})
    print(tokenizer.get_added_vocab())

    article = ["UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."] * 2
    summary = ["Weiter Verhandlung in Syrien."] * 2

    inputs = tokenizer(article, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summary, max_length=256, truncation=True, padding='max_length', return_tensors='pt')

    param_optimizer = list(model.named_parameters())
    # print("---", param_optimizer)

    # outputs = model(inputs, labels)
    outputs = model(**inputs, labels=labels["input_ids"])
    print(outputs.logits.shape)
    loss = outputs[0]
    loss.backward()
    print(loss)


def t5_demo_test():
    mt5_pretrain = "/data/Learn_Project/Backup_Data/mt5-small"
    t5_model = MT5ForConditionalGeneration.from_pretrained(mt5_pretrain)
    # my_model = MultiClassT5(t5_model)

    tokenizer = T5Tokenizer.from_pretrained(mt5_pretrain)
    article = ["UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."] * 2
    summary = ["Weiter Verhandlung in Syrien."] * 2

    inputs = tokenizer(article, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summary, max_length=256, truncation=True, padding='max_length', return_tensors='pt')

    # loss = model(inputs, labels)
    out = t5_model(**inputs, labels=labels["input_ids"])[0]
    print(inputs)
    # out = tokenizer.batch_decode(my_model.generate(inputs))
    # loss.backward()
    print(out)


if __name__ == '__main__':
    # test_report()
    # t5_demo_test()
    # s = filter_special_token("<pad> finance society</s> <pad>")
    t5_demo_generate()