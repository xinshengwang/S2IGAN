data_path=data/102flowers/Oxford102/images_npy
gen_dir=outputs/gan/flowers/TestImage
exp_dir=outputs/gan/flowers

python3 mAP_evaluate.py --data_dir $data_path \
				--gen_dir $gen_dir \
              --exp_dir $exp_dir \
			  
			