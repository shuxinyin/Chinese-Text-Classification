## 1. Text Classification

本项目是基于pytorch,关于一些文本分类任务的一些模型(bert、roberta、T5、textCNN...)实现及笔记，持续更新.

**数据集: [THUCNews]（http://thuctc.thunlp.org）**  ,本项目只选取了部分短文本数据集，已上传，data目录下。
> 数据：10个类  
> label_dic = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5, 'politics': 6,
'sports': 7, 'game': 8, 'entertainment': 9}

#### How To Run

> you can find the code and the illustration in directory of **bert_classification**  
> **不同模型结果的详细结果报告、运行说明等，详见目录bert_classification/README.md。**

### Result

| model         | f1     | Loss   | time(s)/epoch | pretrain-model-download-link                                                                                                             | algorithm nodes                                                            |
|---------------|--------|--------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Albert-tiny   | 0.8011 | 0.6149 | 10.4034       | [albert_chinese](https://huggingface.co/ckiplab/albert-tiny-chinese/tree/main)                                                           | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |
| bert-base     | 0.9102 | 0.3382 | 41.5280       | [bert_chinese](https://huggingface.co/bert-base-chinese/tree/main)                                                                       | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |
| roberta       | 0.9022 | 0.4267 | 45.9066       | [roberta_chinese_L12](https://pan.baidu.com/s/1AGC76N7pZOzWuo8ua1AZfw) <br/> [brightmart_link](https://github.com/brightmart/roberta_zh) | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |  
| macbert-base  | 0.9157 | 0.2811 | 41.4945       | [macbert_chinese](https://huggingface.co/hfl/chinese-macbert-base/tree/main)                                                             | [预训练模型：从MacBERT、SpanBERT看MLM任务](https://zhuanlan.zhihu.com/p/517979209)    |
| spanbert-base | 0.6199 | 0.2811 | 40.9721       | [spanbert_english](https://huggingface.co/SpanBERT/spanbert-base-cased/tree/main)                                                        | [预训练模型：从MacBERT、SpanBERT看MLM任务](https://zhuanlan.zhihu.com/p/517979209)    |
| mT5-small     | 0.8544 | ~      | 73.6844       | [mt5-small](https://huggingface.co/google/mt5-small/tree/main)                                                                           | ~                                                                          |
| TextCNN       | 0.8773 | 0.6357 | 12.8052       | None                                                                                                                                     | [搭一个TextCNN-文本分类利器](https://zhuanlan.zhihu.com/p/386614000)                |

## 2. BERT的几种fine-tune策略

主要实验了fine-tune阶段的几种策略，包括weight_decay、不同的初始化策略、冻结参数、warmup等。

#### 实验中涉及的策略：

> 1. bert_adam
> 2. weight_decay
> 3. reinit pooler
> 4. reinit layers
> 5. frozen parameters
> 6. warmup_proportion

#### Notes:

> a detailed description is given here.     
> 详细笔记: [Bert在fine-tune时训练的几种技巧](https://zhuanlan.zhihu.com/p/524036087)

#### How To Run

> gpu device: 3090ti  
> you can find the code and scripts in directory of **finetune_bert_better**

### Result

本项目主要做了关于以上参数的实验，组合下来，总计64组，所有的实验结果可以在目录下**finetune_bert_better/README.md**找到。
> 其中：1/0 represented use or not  
> 表现最好的为50组f1=0.9405，27epoch*
> 65.4266s（bert_adam1-weight_decay1-reinit_pooler0-reinit_layers0-frozen_layers0-warmup_proportion0.1）。  
> 表现最差的为55组f1=0.9281，28epoch*
> 41.4246s（bert_adam1-weight_decay1-reinit_pooler0-reinit_layers6-frozen_layers6-warmup_proportion0.0）。  
> 加了frozen-parameter的速度会快不少，平均41.4246s/epoch

| index | bertadam | weight_decay | reinit_pooler | reinit_layers | frozen_layers | warmup_proportion | f1         | 
|-------|----------|--------------|---------------|---------------|---------------|-------------------|------------|
| 35    | 1        | 0            | 0             | 0             | 6             | 0.0               | 0.9398     | 
| 37    | 1        | 0            | 0             | 6             | 0             | 0.0               | 0.9287     |
| 45    | 1        | 0            | 1             | 6             | 0             | 0.0               | 0.9284     | 
| 50    | 1        | 1            | 0             | 0             | 0             | 0.1               | **0.9405** | 
| 55    | 1        | 1            | 0             | 6             | 6             | 0.0               | _0.9281_   | 
| 58    | 1        | 1            | 1             | 0             | 0             | 0.1               | 0.9396     | 
