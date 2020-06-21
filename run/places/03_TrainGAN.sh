data_path=data/places/7classes
save_root=outputs/gan/places
cfg_file=cfg/places_3stages.yml
seed=200

python3 main.py --data_dir $data_path \
              --save_root $save_root \
			  --cfg $cfg_file\
			  --manualSeed $seed\
			