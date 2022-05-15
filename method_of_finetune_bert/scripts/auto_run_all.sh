set -x
cd ..

# data path
data_dir="../data/THUCNews/news"
save_path="../ckpt/albert_classification"

# BERT parameter setup (change here while changing pre-train model)
pretrained_path="/data/Learn_Project/Backup_Data/bert_chinese"
#pretrained_path: /data/Learn_Project/Backup_Data/RoBERTa_zh_Large_PyTorch

for warmup_proportion in 0.0 0.1; do
for bertadam in 0 1; do
for weight_decay in 0 1; do
for reinit_pooler in 0 1; do
for reinit_layers in 0 6; do
for freeze_layer_count in 0 6; do


  python train_source.py \
    --data_dir  ${data_dir} \
    --save_path  ${save_path} \
    --pretrained_path ${pretrained_path} \
    --gpu "0" \
    --epoch 32 \
    --lr 5e-5 \
    --warmup_proportion ${warmup_proportion} \
    --pooling_type "first-last-avg" \
    --batch_size 256 \
    --sent_max_len  44 \
    --bertadam ${bertadam} \
    --weight_decay ${weight_decay} \
    --reinit_pooler ${reinit_pooler} \
    --reinit_layers ${reinit_layers} \
    --freeze_layer_count ${freeze_layer_count}


done
done
done
done
done
done
