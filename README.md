## 1. Text Classification

本项目是基于pytorch,关于一些文本分类任务的一些模型实现及笔记，持续更新.

**数据集: [THUCNews]（http://thuctc.thunlp.org）**  ,本项目只选取了部分短文本数据集，已上传，data目录下。  
>数据：10个类  
>label_dic = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5, 'politics': 6,
'sports': 7, 'game': 8, 'entertainment': 9}

#### How To Run
> you can find the code and the illustration in directory of **bert_classification**  
> **不同模型结果的详细结果报告、运行说明等，详见目录bert_classification/README.md。**
 
### Result

| model         | f1     | Loss   | time(s)/epoch | pretrain-model-download-link                                                                                                             | algorithm nodes                                                            |
|---------------|--------|--------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Albert-tiny   | 0.8011 | 0.6149 | 16.4034       | [albert_chinese](https://huggingface.co/ckiplab/albert-tiny-chinese/tree/main)                                                           | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |
| bert-base     | 0.9102 | 0.3382 | 137.7193      | [bert_chinese](https://huggingface.co/bert-base-chinese/tree/main)                                                                       | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |
| roberta       | 0.9022 | 0.4267 | 140.9066      | [roberta_chinese_L12](https://pan.baidu.com/s/1AGC76N7pZOzWuo8ua1AZfw) <br/> [brightmart_link](https://github.com/brightmart/roberta_zh) | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |  
| macbert-base  | 0.9157 | 0.2811 | 137.4945      | [macbert_chinese](https://huggingface.co/hfl/chinese-macbert-base/tree/main)                                                             | [预训练模型：从MacBERT、SpanBERT看MLM任务](https://zhuanlan.zhihu.com/p/517979209)    |
| spanbert-base | 0.6199 | 0.2811 | 131.9721      | [spanbert_english](https://huggingface.co/SpanBERT/spanbert-base-cased/tree/main)                                                        | [预训练模型：从MacBERT、SpanBERT看MLM任务](https://zhuanlan.zhihu.com/p/517979209)    |
| TextCNN       | 0.8773 | 0.6357 | 28.8052       | None                                                                                                                                     | [搭一个TextCNN-文本分类利器](https://zhuanlan.zhihu.com/p/386614000)                |


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
> 详细笔记: [如何让Bert在finetune时更好更稳一点](https://www.zhihu.com/people/da-mo-wang-dare/posts)

#### How To Run
> gpu device: 3090ti
> you can find the code and scripts in directory of **finetune_bert_better**


### Result
本项目主要做了关于以上参数的实验，组合下来，总计64组，所有的实验结果可以在目录下**finetune_bert_better/README.md**找到。
> 其中：1/0 represented use or not  
> 表现最好的为50组f1=0.9405（bert_adam1-weight_decay1-reinit_pooler0-reinit_layers0-frozen_layers0-warmup_proportion0.1）。  
> 表现最差的为50组f1=0.9281（bert_adam1-weight_decay1-reinit_pooler0-reinit_layers6-frozen_layers6-warmup_proportion0.0）。 

| index | bertadam | weight_decay | reinit_pooler | reinit_layers | frozen_layers | warmup_proportion | f1         | 
|-------|----------|--------------|---------------|---------------|---------------|-------------------|------------|
| 1     | 0        | 0            | 0             | 0             | 0             | 0.0               | 0.9376     |     
| 2     | 0        | 0            | 0             | 0             | 0             | 0.1               | 0.9379     |     
| 3     | 0        | 0            | 0             | 0             | 6             | 0.0               | 0.9358     |     
| 4     | 0        | 0            | 0             | 0             | 6             | 0.1               | 0.9372     |     
| 5     | 0        | 0            | 0             | 6             | 0             | 0.0               | 0.9359     |     
| 6     | 0        | 0            | 0             | 6             | 0             | 0.1               | 0.9348     |     
| 7     | 0        | 0            | 0             | 6             | 6             | 0.0               | 0.9315     |     
| 8     | 0        | 0            | 0             | 6             | 6             | 0.1               | 0.9333     |     
| 9     | 0        | 0            | 1             | 0             | 0             | 0.0               | 0.939      |     
| 10    | 0        | 0            | 1             | 0             | 0             | 0.1               | 0.9386     |     
| 11    | 0        | 0            | 1             | 0             | 6             | 0.0               | 0.937      |     
| 12    | 0        | 0            | 1             | 0             | 6             | 0.1               | 0.9357     |     
| 13    | 0        | 0            | 1             | 6             | 0             | 0.0               | 0.9344     |     
| 14    | 0        | 0            | 1             | 6             | 0             | 0.1               | 0.9337     |     
| 15    | 0        | 0            | 1             | 6             | 6             | 0.0               | 0.932      |     
| 16    | 0        | 0            | 1             | 6             | 6             | 0.1               | 0.9315     |     
| 17    | 0        | 1            | 0             | 0             | 0             | 0.0               | 0.9395     |     
| 18    | 0        | 1            | 0             | 0             | 0             | 0.1               | 0.9394     |     
| 19    | 0        | 1            | 0             | 0             | 6             | 0.0               | 0.9391     |     
| 20    | 0        | 1            | 0             | 0             | 6             | 0.1               | 0.9364     |     
| 21    | 0        | 1            | 0             | 6             | 0             | 0.0               | 0.9357     |     
| 22    | 0        | 1            | 0             | 6             | 0             | 0.1               | 0.9357     |     
| 23    | 0        | 1            | 0             | 6             | 6             | 0.0               | 0.9334     |     
| 24    | 0        | 1            | 0             | 6             | 6             | 0.1               | 0.9327     |     
| 25    | 0        | 1            | 1             | 0             | 0             | 0.0               | 0.9393     |     
| 26    | 0        | 1            | 1             | 0             | 0             | 0.1               | 0.9375     |     
| 27    | 0        | 1            | 1             | 0             | 6             | 0.0               | 0.937      |     
| 28    | 0        | 1            | 1             | 0             | 6             | 0.1               | 0.938      |     
| 29    | 0        | 1            | 1             | 6             | 0             | 0.0               | 0.9342     |     
| 30    | 0        | 1            | 1             | 6             | 0             | 0.1               | 0.9357     |     
| 31    | 0        | 1            | 1             | 6             | 6             | 0.0               | 0.9321     |     
| 32    | 0        | 1            | 1             | 6             | 6             | 0.1               | 0.9307     |     
| 33    | 1        | 0            | 0             | 0             | 0             | 0.0               | 0.9361     | 
| 34    | 1        | 0            | 0             | 0             | 0             | 0.1               | 0.9379     | 
| 35    | 1        | 0            | 0             | 0             | 6             | 0.0               | 0.9398     | 
| 36    | 1        | 0            | 0             | 0             | 6             | 0.1               | 0.937      | 
| 37    | 1        | 0            | 0             | 6             | 0             | 0.0               | 0.9287     | 
| 38    | 1        | 0            | 0             | 6             | 0             | 0.1               | 0.9351     | 
| 39    | 1        | 0            | 0             | 6             | 6             | 0.0               | 0.9312     | 
| 40    | 1        | 0            | 0             | 6             | 6             | 0.1               | 0.9306     | 
| 41    | 1        | 0            | 1             | 0             | 0             | 0.0               | 0.9342     | 
| 42    | 1        | 0            | 1             | 0             | 0             | 0.1               | 0.9371     | 
| 43    | 1        | 0            | 1             | 0             | 6             | 0.0               | 0.9376     | 
| 44    | 1        | 0            | 1             | 0             | 6             | 0.1               | 0.9391     | 
| 45    | 1        | 0            | 1             | 6             | 0             | 0.0               | 0.9284     | 
| 46    | 1        | 0            | 1             | 6             | 0             | 0.1               | 0.9349     | 
| 47    | 1        | 0            | 1             | 6             | 6             | 0.0               | 0.9304     | 
| 48    | 1        | 0            | 1             | 6             | 6             | 0.1               | 0.9314     | 
| 49    | 1        | 1            | 0             | 0             | 0             | 0.0               | 0.9377     | 
| 50    | 1        | 1            | 0             | 0             | 0             | 0.1               | **0.9405** | 
| 51    | 1        | 1            | 0             | 0             | 6             | 0.0               | 0.9394     | 
| 52    | 1        | 1            | 0             | 0             | 6             | 0.1               | 0.9372     | 
| 53    | 1        | 1            | 0             | 6             | 0             | 0.0               | 0.9304     | 
| 54    | 1        | 1            | 0             | 6             | 0             | 0.1               | 0.935      | 
| 55    | 1        | 1            | 0             | 6             | 6             | 0.0               | _0.9281_   | 
| 56    | 1        | 1            | 0             | 6             | 6             | 0.1               | 0.9328     | 
| 57    | 1        | 1            | 1             | 0             | 0             | 0.0               | 0.9348     | 
| 58    | 1        | 1            | 1             | 0             | 0             | 0.1               | 0.9396     | 
| 59    | 1        | 1            | 1             | 0             | 6             | 0.0               | 0.9391     | 
| 60    | 1        | 1            | 1             | 0             | 6             | 0.1               | 0.9355     | 
| 61    | 1        | 1            | 1             | 6             | 0             | 0.0               | 0.9301     | 
| 62    | 1        | 1            | 1             | 6             | 0             | 0.1               | 0.9365     | 
| 63    | 1        | 1            | 1             | 6             | 6             | 0.0               | 0.9304     | 
| 64    | 1        | 1            | 1             | 6             | 6             | 0.1               | 0.9332     | 