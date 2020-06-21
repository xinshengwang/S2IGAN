real_path=data/birds/CUB_200_2011/Test_image
gen_path=outputs/gan/birds/TestImage
save_path=outputs/gan/birds/FID.text

python3 fid_score.py --real_path $real_path \
                --gen_path $gen_path \
				--save_path $save_path
              
			