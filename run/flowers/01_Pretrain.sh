data_path=data/102flowers
save_root=outputs/pre_train/flowers
cfg_file=cfg/Pretrain/flower_train.yml
seed=200
lr=0.001
wd=1e-3
batch=128

python3 pretrain_speechembedding.py --data_path $data_path \
              --save_root $save_root \
			  --cfg_file $cfg_file\
			  --lr $lr\
			  --manualSeed $seed\
			  --weight-decay $wd \
			  --batch-size $batch