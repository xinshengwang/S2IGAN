# S2IGAN
This is the pytorch implement for our paper [S2IGAN: Speech-to-Image Generation via Adversarial Learning](https://arxiv.org/abs/2005.06968). More results can be seen in the [project page](https://xinshengwang.github.io/project/s2igan/).

### Data processing
#### CUB-200 (Bird) and Oxford-102 (Flower)
**step0:** You can download the synthesized spoken caption database from the [project page](https://xinshengwang.github.io/project/s2igan/), and then go to "step3". Or, you can start from "step 1" and synthesize spoken caption by yourself.

**step1:** Download CUB and Oxford [Image and Text Captions](https://github.com/reedscot/icml2016)

**step2:** Using a TTS system to transfer text captions to speech captions. In our work, Tacotron2 pre-trained by [NVIDIA](https://github.com/NVIDIA/tacotron2) was adopted. The original code released by [NVIDIA](https://github.com/NVIDIA/tacotron2) doesn't provide the code for the batch inference. We made slight changes to make it can be used to perform batch inference directly. You can download it [here](https://github.com/xinshengwang/Tacotron2_batch_inference).

**step3:** To speed up the training process, we transferred the wav audio to filter bank spectrogram in advance.
```
python data_processing/Audio_to_mel.py
```

**step4:** Download the train/test split files for [CUB](https://drive.google.com/drive/folders/1HQEHjQht33e9STuFR938bwxskA22r6E7?usp=sharing) and [Oxford](https://drive.google.com/drive/folders/1cjGIxIVEK6YNfSTk1P5tZEm3vPFoPSjO?usp=sharing)


**Directory tree**

```
├── birds
│   ├── CUB_200_2011
│   │   ├── auido
│   │   ├── audio_mel
│   │   ├── images
│   ├── train
│   │   ├── filenames.pickle
│   │   ├── class_info.pickle
│   ├── test
│   │   ├── filenames.pickle
│   │   ├── class_info.pickle
```

```
├── flowers
│   ├── Oxford102
│   │   ├── auido
│   │   ├── audio_mel
│   │   ├── images
│   ├── train
│   │   ├── filenames.pickle
│   │   ├── class_info.pickle
│   ├── test
│   │   ├── filenames.pickle
│   │   ├── class_info.pickle
```

#### Places-subset
Download [Places audio data](https://groups.csail.mit.edu/sls/downloads/placesaudio/downloads.cgi). Images of Places-subset and split files can be downloaded [here](https://drive.google.com/drive/folders/1yofQsOlOceOgMAav2ssd4N3Vn9GoyC0Z?usp=sharing). Database files are organized as follows

```
├── places
│   ├── images
│   ├── audio
│   │   ├── mel
│   │   ├── wav
│   ├── train
│   │   ├── filenames.pickle
│   ├── test
│   │   ├── filenames.pickle
```

#### Flickr8k
**step1:** Download Flickr8k [Image](http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b) and [Audio Captions](https://groups.csail.mit.edu/sls/downloads/flickraudio/index.cgi)

**step2:** Transfer wav to spectrogram.

**step2:** Download the[ train/test split files](https://drive.google.com/drive/folders/1TTv8bQBbus8xUexXkQnMMrdHa3UvwqWS?usp=sharing).


```
├── Flickr8k
│   ├── images
│   ├── flickr_audio
│   │   ├── mel
│   │   ├── wavs
│   ├── train
│   │   ├── filenames.pickle
│   ├── test
│   │   ├── filenames.pickle
```

### Running Step-by-step
Note: Change the path in .sh files to your data path. If you use the speech embedding provided by us (see Step 2), you can start from **step3**. 

**step1:** Train SEN

```
sh run/flickr/01_Pretrain.sh
```


**step2:** Extract speech embeddings

```
sh run/flickr/02_Extracting.sh
```
You can skip the first two steps by using our provided speech embeddings and pre-trained image encoder for [CUB](https://drive.google.com/drive/folders/1tICN6_DrkzZu-pB6Z7c5Q2uomvHWYPsg?usp=sharing), [Oxford](https://drive.google.com/drive/folders/1u1Kn-79Kldqh342sapayML1Hwi6VVY74?usp=sharing), [Flickr8k](https://drive.google.com/drive/folders/1lXxwJQ07rFJM-vaTQMfLPyazSj6O_-Os?usp=sharing), and [Places-subset](https://drive.google.com/drive/folders/1bVLQug8gtSYol097TBxYakdCgarHCooH?usp=sharing). Then put these embeddings as follows:

```
├── outputs
│   ├── pre_train
│   │   ├── birds
│   │   │   ├── speech_embeddings_train.pickle
│   │   │   ├── speech_embeddings_test.pickle
│   │   │   ├── models
│   │   │   │   ├── best_image_model.pth 
│   │   ├── flowers
│   │   │   ├── speech_embeddings_train.pickle
│   │   │   ├── speech_embeddings_test.pickle
│   │   │   ├── models
│   │   │   │   ├── best_image_model.pth
│   │   ├── flickr
│   │   │   ├── speech_embeddings_train.pickle
│   │   │   ├── speech_embeddings_test.pickle
│   │   │   ├── models
│   │   │   │   ├── best_image_model.pth
│   │   ├── places
│   │   │   ├── speech_embeddings_train.pickle
│   │   │   ├── speech_embeddings_test.pickle
│   │   │   ├── models
│   │   │   │   ├── best_image_model.pth
```

**step3:** Train the generator
```
sh run/flickr/03_TrainGAN.sh
```

**step4:** Generate images
```
sh run/flickr/04_GenImage.sh
```

**step5:** Calculate Insception Score (IS)

For Flickr and Places-subset, you can directly run the .sh files in the corresponding directory, such as
```
sh run/flickr/05_InsceptionScore_generally.sh
```

For CUB and Oxford, we use the [fine-tuned model](https://github.com/egvincent/styled-stackgan/tree/e496e10873666bce5de39bb2d41186546af31f64/StackGAN-inception-model)

**step6:** Semantic Consistency Evaluation

For Flickr and Places-subset:

```
sh run/flickr/06_Recall.sh
```

For CUB and Oxford:

```
sh run/birds/06_mAP.sh
```
**step7:** FID 
Download the [code](https://github.com/mseitzer/pytorch-fid) to calculate the FID

### Cite
```
@article{wang2020s2igan,
  title={S2IGAN: Speech-to-Image Generation via Adversarial Learning},
  author={Wang, Xinsheng and Qiao, Tingting and Zhu, Jihua and Hanjalic, Alan and Scharenborg, Odette},
  journal={arXiv preprint arXiv:2005.06968},
  year={2020}
}
```
