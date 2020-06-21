data_path=data/birds/CUB_200_2011/images_npy
gen_dir=outputs/gan/birds/TestImage
exp_dir=outputs/gan/birds

python3 mAP_evaluate.py --data_dir $data_path \
				--gen_dir $gen_dir \
              --exp_dir $exp_dir \
			  
			