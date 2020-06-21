data_path=data/Flickr8k
gen_dir=outputs/gan/flickr/TestImage
exp_dir=outputs/gan/flickr

python3 Recall_evaluate.py --data_dir $data_path \
				--gen_dir $gen_dir \
              --exp_dir $exp_dir \
			  
			