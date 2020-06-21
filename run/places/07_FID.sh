real_path=data/places/7classes/Test_image
gen_path=outputs/gan/places/TestImage
save_path=outputs/gan/places/FID.text

python3 fid_score.py --real_path $real_path \
                --gen_path $gen_path \
				--save_path $save_path
              
			