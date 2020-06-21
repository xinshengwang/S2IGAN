data_path=data/birds
save_root=outputs/gan/birds
cfg_file=cfg/birds_3stages.yml
seed=200

python3 main.py --data_dir $data_path \
              --save_root $save_root \
			  --cfg $cfg_file\
			  --manualSeed $seed\
			