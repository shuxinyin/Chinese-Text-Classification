## Text Classification（pytorch）

本项目是关于一些文本分类任务的一些模型实现及笔记，持续更新.

**数据集: [THUCNews]（http://thuctc.thunlp.org/）**  ,本项目只选取了部分短文本数据集，已上传，data目录下。  
数据：10个类  
{'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}

### Result

| model        | f1     | Loss   | time(s)/epoch | pretrain-model-download-link                                                                                                             | algorithm nodes                                                            |
|--------------|--------|--------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Albert-tiny  | 0.8011 | 0.6149 | 16.4034       | [albert_chinese](https://huggingface.co/ckiplab/albert-tiny-chinese/tree/main)                                                           | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |
| bert-base    | 0.9102 | 0.3382 | 137.7193      | [bert_chinese](https://huggingface.co/bert-base-chinese/tree/main)                                                                       | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |
| roberta      | 0.9022 | 0.4267 | 140.9066      | [roberta_chinese_L12](https://pan.baidu.com/s/1AGC76N7pZOzWuo8ua1AZfw) <br/> [brightmart_link](https://github.com/brightmart/roberta_zh) | [预训练模型：从BERT到XLNet、RoBERTa、ALBERT](https://zhuanlan.zhihu.com/p/436017910) |  
| macbert-base | 0.9157 | 0.2811 | 137.4945      | [macbert_chinese](https://huggingface.co/hfl/chinese-macbert-base/tree/main)                                                             | wait to update                                                             |
| TextCNN      | 0.8773 | 0.6357 | 28.8052       | None                                                                                                                                     | [搭一个TextCNN-文本分类利器](https://zhuanlan.zhihu.com/p/386614000)                |

**不同模型结果的详细结果报告、如何运行等，详见单个项目内README.md。**

