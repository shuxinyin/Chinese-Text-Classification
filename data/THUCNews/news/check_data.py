import random
import pandas as pd



train = pd.read_csv("./train.txt", header=None, sep='\t', names=["text", "label"])
test = pd.read_csv("./test.txt", header=None, sep='\t', names=["text", "label"])
valid = pd.read_csv("./dev.txt", header=None, sep='\t', names=["text", "label"])
print(train.shape, test.shape, valid.shape)

train["text_len"] = train.text.map(len)
test["text_len"] = train.text.map(len)
valid["text_len"] = train.text.map(len)
print(train.describe())
print(test.describe())
print(valid.describe())

data_weight_dic = {'房产': 0.15, '教育': 0.7,  '时政': 0.2, '游戏': 1, '科技': 1, '财经': 0.1}

# df = pd.concat([train, test], axis=0)

# train_data_list, valid_data_list = list(), list()
# for k, v in data_weight_dic.items():
#     df_tmp = df[df.label == k].sample(frac=v, replace=False, random_state=1)
#     df_tmp_valid = valid[valid.label == k].sample(frac=1, replace=False, random_state=1)
#     train_data_list.append(df_tmp)
#     valid_data_list.append(df_tmp_valid)



# df_train = pd.concat(train_data_list, axis=0)
# df_valid = pd.concat(valid_data_list, axis=0)
# print(df_train.shape, df_valid.shape)
# print(df_train["label"].value_counts())

# df_train.to_csv("./unbalanced_THUCNews/cnews.train.txt", index=False, header=None, sep='\t')
# df_valid.to_csv("./unbalanced_THUCNews/cnews.val.txt", index=False, header=None, sep='\t')

# df.to_csv(("./unbalanced_THUCNews/cnews.train.txt", index=False, header=None, sep='\t')

