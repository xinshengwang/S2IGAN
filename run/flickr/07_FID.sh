real_path=data/Flickr8k/test_Image
gen_path=outputs/gan/flickr/TestImage
save_path=outputs/gan/flickr/FID.text

python3 fid_score.py --real_path $real_path \
                --gen_path $gen_path \
				--save_path $save_path
              
			