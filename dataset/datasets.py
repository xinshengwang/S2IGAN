from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random
import numpy as np
import pandas as pd
import six
import sys
# import pdb

from utils.config import cfg
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def normalizeFeature(x):	
    
    # x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    # feature_norm =  ((x**2).sum())**0.5           #       np.sum(x**2, axis=1)**0.5 # l2-norm
    # feat = x / feature_norm
    x_max = x.max()
    x_min = x.min()
    feat = (x-x_min)/(x_max-x_min)
    return feat


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


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

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

def get_single_img(img_path, imsize, bbox=None, transform=None, normalize=None):
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



class ImageFolder(data.Dataset):
    def __init__(self, root, split_dir='train', custom_classes=None,
                 base_size=64, transform=None, target_transform=None):
        root = os.path.join(root, split_dir)
        classes, class_to_idx = self.find_classes(root, custom_classes)
        imgs = self.make_dataset(classes, class_to_idx)
        if imgs:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        print('num_classes', self.num_classes)

    def find_classes(self, directory, custom_classes):
        classes = []

        for d in os.listdir(directory):
            if os.path.isdir:
                if custom_classes is None or d in custom_classes:
                    classes.append(os.path.join(directory, d))
        print('Valid classes: ', len(classes), classes)

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, classes, class_to_idx):
        images = []
        for d in classes:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[d])
                        images.append(item)
        print('The number of images: ', len(images))
        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        imgs_list = get_imgs(path, self.imsize, transform=self.transform, normalize=self.norm)
        return imgs_list

    def __len__(self):
        return len(self.imgs)


class SpeechDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='melspec',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        for i, item in enumerate(filenames): # this is the range of the number of images
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_embedding(self, data_dir, embedding_type):
          
        if data_dir.find('birds') != -1:
            if cfg.TRAIN.FLAG:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'birds' + '/' + 'speech_embeddings_train.pickle'
            else:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'birds' + '/'  'speech_embeddings_test.pickle'                
        elif data_dir.find('flowers') != -1:
            if cfg.TRAIN.FLAG:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'flowers' + '/' + 'speech_embeddings_train.pickle'
            else:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'flowers' + '/' + 'speech_embeddings_test.pickle'
        elif data_dir.find('Flickr8k') != -1:
            if cfg.TRAIN.FLAG:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'flickr' + '/' + 'speech_embeddings_train.pickle'
            else:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'flickr' + '/' + 'speech_embeddings_test.pickle'
        elif data_dir.find('places') != -1:
            if cfg.TRAIN.FLAG:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'places' + '/' + 'speech_embeddings_train.pickle'
            else:
                embedding_filename = 'outputs' + '/' + 'pre_train' + '/' + 'places' + '/' + 'speech_embeddings_test.pickle'
        else:
            raise Exception("wrong data path")
        

        if embedding_type != 'Audio_emb':
            with open(data_dir + embedding_filename, 'rb') as f:
                embeddings = pickle.load(f, encoding="bytes")
                # pdb.set_trace()
                embeddings = np.array(embeddings)
                # if len(embeddings.shape)==2:
                #     embeddings = embeddings[:,np.newaxis,:]
        else:
            with open(embedding_filename, 'rb') as f:
                embeddings = pickle.load(f, encoding="bytes")
                embeddings = np.array(embeddings)

            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        data_dir = self.data_dir
        key = self.filenames[index]
        class_id = self.class_id[index]
        self.class_id = np.array(self.class_id)
        if data_dir.find('birds') != -1 or data_dir.find('flowers') != -1:
            same_indexs = np.where(self.class_id==class_id)[0] 
            same_indexs = list(set(same_indexs)-set([index]))
            same_index = random.choice(same_indexs)
        else:
            same_index = index
        
        if data_dir.find('birds') != -1:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        elif data_dir.find('flowers') != -1:
            bbox = None
            data_dir = '%s/Oxford102' % self.data_dir
        elif data_dir.find('Flickr8k') != -1 or data_dir.find('places') != -1:
            bbox = None
            data_dir = self.data_dir
        else:
            raise Exception('wrong data path')
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if self.class_id[index] == self.class_id[wrong_ix]:
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        same_key = self.filenames[same_index]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
            same_bbox = self.bbox[same_key]
        else:
            wrong_bbox = None
            same_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % (data_dir, wrong_key)
        same_image_name = '%s/images/%s.jpg' % (data_dir, same_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize, wrong_bbox, self.transform, normalize=self.norm)
        same_imgs = get_single_img(same_image_name, self.imsize, same_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]

        
        if cfg.EMBEDDING_TYPE =='melspec':
            embedding = normalizeFeature(embedding)
        
        
        
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs,same_imgs, embedding, class_id, key  # captions

    def prepair_test_pairs(self, index):
        data_dir = self.data_dir
        key = self.filenames[index]
        if data_dir.find('birds') != -1:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        elif data_dir.find('flowers') != -1:
            bbox = None
            data_dir = '%s/Oxford102' % self.data_dir
        elif data_dir.find('Flickr8k') != -1 or data_dir.find('places'):
            bbox = None
            data_dir = self.data_dir
        else:
            raise Exception('wrong data path')
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]   #[index,:,:]  changed by shawn


        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)

        if cfg.EMBEDDING_TYPE =='melspec':
            embeddings = normalizeFeature(embeddings)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)

