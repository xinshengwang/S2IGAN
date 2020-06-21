data_path=data/birds
save_root=outputs/gan/birds
cfg_file=cfg/eval_birds.yml
seed=100

python3 main.py --data_dir $data_path \
              --save_root $save_root \
			  --cfg $cfg_file\
			  --manualSeed $seed\
			