checkpoint_dir=/StackGAN-inception-model-master/inception_finetuned_models/birds_valid299/model.ckpt
image_folder=/outputs/gan/birds/TestImage

python3 inception_score.py --checkpoint_dir $checkpoint_dir \
                --image_folder $image_folder
              
			