from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time

from cfg.Pretrain.config import cfg
from torch.utils.data.dataloader import default_collate


import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt


import os
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def prepare_data(data):    
    imgs, captions, class_ids, keys, spec_length = data  
    
    if cfg.CUDA:
        real_imgs = (Variable(imgs).cuda())
        captions = Variable(captions).cuda()
        spec_length = Variable(spec_length).cuda()
       
    else:
        real_imgs = Variable(imgs)
        captions = Variable(captions)    
        spec_length = Variable(spec_length)
    
    return real_imgs, captions,class_ids, keys, spec_length


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')
    if cfg.TRAIN.MODAL != 'extraction':
        for elem in batch:
            if cfg.TRAIN.MODAL != 'extraction':
                imgs, caps, cls_id, key, label = elem
            max_input_len = max_input_len if max_input_len > caps.shape[0] else caps.shape[0]       

        for i, elem in enumerate(batch):
            imgs, caps, cls_id, key,label = elem
            input_length = caps.shape[0]
            input_dim = caps.shape[1]
            # print('f.shape: ' + str(f.shape))
            feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            feature[:caps.shape[0], :caps.shape[1]] = caps       
            
            batch[i] = (imgs, feature, cls_id, key, input_length, label)
            # print('feature.shape: ' + str(feature.shape))
            # print('trn.shape: ' + str(trn.shape))

        batch.sort(key=lambda x: x[-2], reverse=True)

    return default_collate(batch)

def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    return normalize(img)

def get_audio(input_file):
    y, sr = librosa.load(input_file, sr=None)

    if cfg.add_noise:
        noise = np.random.randn(len(y))
        y = y + 0.01 * noise
    ws = int(sr * 0.001 * cfg.SPEECH.window_size)
    st = int(sr * 0.001 * cfg.SPEECH.stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = cfg.SPEECH.input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if cfg.cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')  


# take this when the SPEECH.model = cnn
# input of SPEECH is wav
def get_audio_for_cnn(path):
    audio_type = 'melspectrogram'
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
    preemph_coef = 0.97
    sample_rate = 22050 # original given by 16000
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    target_length = 2048
    use_raw_length = False
    padval =  0
    fmin =  20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y, sr = librosa.load(path, sample_rate)
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    # y = preemphasis(y, preemph_coef)
    # compute mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        window=window_type)
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
        logspec = librosa.power_to_db(spec, ref=np.max)
    n_frames = logspec.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
            constant_values=(padval,padval))
    elif p < 0:
        logspec = logspec[:,0:p]
        n_frames = target_length
    # logspec = torch.FloatTensor(logspec)
    return np.swapaxes(logspec, 0, 1).astype('float32')    #,logspec 

#when the SPEECH.styple = 'npy'
def audio_processing(input_file):
    
    y = input_file
    sr = cfg.SPEECH.sample 
    if cfg.add_noise:
        noise = np.random.randn(len(y))
        y = y + 0.01 * noise
    ws = int(sr * 0.001 * cfg.SPEECH.window_size)
    st = int(sr * 0.001 * cfg.SPEECH.stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = cfg.SPEECH.input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if cfg.cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')  

# when the SPEECH.style ='npy'
# take this when SPEECH.model ='CNN'
def audio_processing_for_cnn(input_file):
    y = input_file
    sr = cfg.SPEECH.sample 
    target_length = 2048
    if cfg.add_noise:
        noise = np.random.randn(len(y))
        y = y + 0.01 * noise
    ws = int(sr * 0.001 * cfg.SPEECH.window_size)
    st = int(sr * 0.001 * cfg.SPEECH.stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = cfg.SPEECH.input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    n_frames = feat.shape[1]
    if target_length > n_frames:
        feat = np.tile(feat,(1,target_length//n_frames+1))
    feat = feat[:,:target_length]

    return feat

"""
def audio_processing_for_cnn(input_file):
    
    audio_type = 'melspectrogram'
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
    preemph_coef = 0.97
    sample_rate = 22050
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    target_length = 2048
    use_raw_length = False
    padval =  0
    fmin =  20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y = input_file
    sr = cfg.SPEECH.sample
    # y, sr = librosa.load(path, sample_rate)

    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    # y = preemphasis(y, preemph_coef)
    # compute mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        window=window_type)
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
        logspec = librosa.power_to_db(spec, ref=np.max)
    n_frames = logspec.shape[1]
    if use_raw_length:
        target_length = n_frames
    
    
    # p = target_length - n_frames
    # if p > 0:
    #     logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
    #         constant_values=(padval,padval))
    # elif p < 0:
    #     logspec = logspec[:,0:p]
    #     n_frames = target_length
    # logspec = torch.FloatTensor(logspec)
   

    if target_length > n_frames:
        logspec = np.tile(logspec,(1,target_length//n_frames+1))
    logspec = logspec[:,:target_length]
    return logspec #, n_frames 
    """

def get_imgs_for_per_train(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                    std = [0.5,0.5,0.5])
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])


    resize = transforms.Resize(355)
    tencrop = transforms.TenCrop(299)
    tens = transforms.ToTensor()

    img = resize(img)
    img = tencrop(img)
    # plt.figure()
    # for i in range(10):
    #     plt.subplot(2,5,i+1)
    #     plt.imshow(img[i])
    # plt.show()


    im = torch.cat([normalize(tens(x)).unsqueeze(0) for x in img])
    im = torch.autograd.Variable(im).cuda()

    if not im.size()[1] == 3:
        im = im.expand(im.size()[0], 3, im.size()[2], im.size()[3])


    return im

# dataloader for the main programer
# used for RNN
class SpeechDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 img_size=64,
                 transform=None, target_transform=None):
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.SPEECH.CAPTIONS_PER_IMAGE   
        self.imsize = img_size
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:   
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)
      
        self.filenames = self.load_filenames(data_dir, split)
        """

        if split == 'train':
            with open('sampled_train.pickle','rb') as f1:
                self.filenames = pickle.load(f1)
        if split=='test':
            with open('sampled_test.pickle','rb') as f2:
                self.filenames = pickle.load(f2)

        # get the class id for the sampled dataset
        self.class_id = []
        for name in (self.filenames):
            num,name = name.split('.')
            num = int(num)
            self.class_id.append(num)
        
        unique_id = sorted(np.unique(self.class_id))
        seq_labels = np.zeros(20)
        for i in range(6):
            seq_labels[unique_id[i]-1]=i
        
        self.labels = seq_labels[np.array(self.class_id)-1]
        """
        
        #load the class_id for the whole dataset
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        
        # cacluate the sequence label for the whole dataset
        if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
            if self.split =='train':
                unique_id = np.unique(self.class_id)
                seq_labels = np.zeros(cfg.DATASET_ALL_CLSS_NUM)
                for i in range(cfg.DATASET_TRAIN_CLSS_NUM):
                    seq_labels[unique_id[i]-1]=i
                
                self.labels = seq_labels[np.array(self.class_id)-1]
        
        if cfg.TRAIN.MODAL != 'extraction' and cfg.TRAIN.MODAL != 'sne':
            if cfg.IMG.style == 'pickle':
                image_path = os.path.join(split_dir,'image_data.pickle')
                with open(image_path,'rb') as f:
                    self.image_data = pickle.load(f)
            if cfg.SPEECH.style == 'pickle':
                audio_path = os.path.join(split_dir,'audio_mel_data.pickle')
                with open(audio_path,'rb') as f:
                    self.audio_data = pickle.load(f)
        # caculate the sequence label for the sampled dataset   
        
      
        
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):      
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames


    def __getitem__(self, index):
        #
        # start = time.time()
        key = self.filenames[index]     #图像名称
        cls_id = self.class_id[index]    
        if self.split =='train':
            if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
                label = self.labels[index]
            else:
                label = cls_id 
        else:
            label = cls_id  #        
        
        if cfg.TRAIN.MODAL != 'extraction':
            if self.data_dir.find('birds') != -1: 
                bbox = self.bbox[key]
                data_dir = '%s/CUB_200_2011' % self.data_dir            
            elif self.data_dir.find('flowers') != -1: 
                bbox = None
                data_dir = '%s/Oxford102' % self.data_dir            
            elif self.data_dir.find('Flickr8k') != -1: 
                bbox = None
                data_dir = self.data_dir
            elif self.data_dir.find('places') != -1: 
                bbox = None
                data_dir = self.data_dir

            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)          

       
        if cfg.SPEECH.style == 'WAV':

            audio_file = '%s/audio/%s' % (data_dir, key)            
            audio_names = os.listdir(audio_file)
            # random select a sentence
            # audio_ix = random.randint(0, self.embeddings_num)    
            if self.split=='train':
                audio_ix = random.randint(0, self.embeddings_num)
            else:
                audio_ix = 0
            audio_name = audio_names[audio_ix]
            audio_path = os.path.join(audio_file, audio_name)
            if cfg.SPEECH.model== 'CNN':
                caps = get_audio_for_cnn(audio_path)
            else:
                caps = get_audio(audio_path)  
        elif cfg.SPEECH.style == 'npy':
            audio_file = '%s/audio_npy/%s.npy' % (data_dir, key) 
            if self.split=='train':
                audio_ix = random.randint(0, self.embeddings_num)
            else:
                audio_ix = 0
            audios = np.load(audio_file,allow_pickle=True)
            audio = audios[audio_ix] 
            if cfg.SPEECH.model== 'CNN':
                caps = audio_processing_for_cnn(audio)
            elif cfg.SPEECH.model == 'CRNN_D':   #CRNN in the work of Danny "language learning using speech to image retrieval"
                pass
            
            else:
                caps = audio_processing(audio)
        elif cfg.SPEECH.style == 'mel':
            if  self.data_dir.find('Flickr8k') != -1:
                audio_file = '%s/flickr_audio/mel/%s.npy' % (data_dir, key) 
            elif self.data_dir.find('places') != -1:
                audio_file = '%s/audio/mel/%s.npy' % (data_dir, key) 
            else:
                audio_file = '%s/audio_mel/%s.npy' % (data_dir, key) 
            
            if self.split=='train':
                audio_ix = random.randint(0, self.embeddings_num)
            else:
                audio_ix = 0            
            audios = np.load(audio_file,allow_pickle=True)
            if len(audios.shape)==2:
                audios = audios[np.newaxis,:,:]            
            
            if cfg.TRAIN.MODAL != 'extraction':
                caps = audios[audio_ix] 
            else:
                caps = audios
        elif cfg.SPEECH.style == 'pickle':
            if self.split=='train':
                audio_ix = random.randint(0, self.embeddings_num)
            else:
                audio_ix = 0
            audios = self.audio_data[key]
            caps = audios[audio_ix]           
            
        
        else:
            print('Error style of audio')
        # end = time.time()
        # duration = end-start
        # print('the duration is = ',duration)
        # if self.split =='train':
        
        if cfg.TRAIN.MODAL =='extraction':
            return caps
        else:
            return imgs, caps, cls_id, key, label   

    def __len__(self):
        return len(self.filenames)

