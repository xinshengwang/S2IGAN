real_path=data/102flowers/Oxford102/Test_image
gen_path=outputs/gan/flowers/TestImage
save_path=outputs/gan/flowers/FID.text

python3 fid_score.py --real_path $real_path \
                --gen_path $gen_path \
				--save_path $save_path
              
			