data_path=data/places/7classes
gen_dir=outputs/gan/places/TestImage
exp_dir=outputs/gan/places

python3 Recall_evaluate.py --data_dir $data_path \
				--gen_dir $gen_dir \
              --exp_dir $exp_dir \
			  
			