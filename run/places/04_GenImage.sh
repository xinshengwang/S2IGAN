data_path=data/places/7classes
save_root=outputs/gan/places
cfg_file=cfg/eval_places.yml
seed=100

python3 main.py --data_dir $data_path \
              --save_root $save_root \
			  --cfg $cfg_file\
			  --manualSeed $seed\
			