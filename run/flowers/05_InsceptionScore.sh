checkpoint_dir=StackGAN-inception-model-master/inception_finetuned_models/flowers_valid299/model.ckpt
image_folder=outputs/gan/flowers/TestImage
save_path=outputs/gan/flowers/inception_score.text
num_classes=20
python3 inception_score.py --checkpoint_dir $checkpoint_dir \
                --image_folder $image_folder \
				--save_path $save_path \
				--num_classes $num_classes
              
			