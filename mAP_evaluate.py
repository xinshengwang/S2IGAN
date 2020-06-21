#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from PIL import Image
from torchvision import models
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
import argparse

def get_imgs(img_path, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')    
    
    if transform is not None:
        img = transform(img)

    if normalize is not None:
        img = normalize(img)
    
    totens = transforms.ToTensor()
    img = totens(img)
    return img

def normalizeFeature(x):	
    
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat

class Inception_v3(nn.Module):
    def __init__(self):
        super(Inception_v3, self).__init__()        

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)       

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

   
    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x,size=(299, 299), mode='bilinear', align_corners=False)  #上采样或者下采样至给定size
        # 299 x 299 x 3

        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        # features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)   # for visual_feature_extraction.py  use this as the output
        x = x.mean(dim=(2,3))
        # 2048

        # global image features
        # cnn_code = self.emb_cnn_code(x)   
        # 512
        # if features is not None:
        #     features = self.emb_features(features)
        return x #cnn_code  #1024


class mAPData(torch.utils.data.Dataset):
    def __init__(self, path,split,image_path):
        self.split = split 
        self.image_path = image_path
        if split == 'gen':
            self.gen_root = image_path
        else:
            self.ground_root = image_path     
        with open(path,'rb') as f:
            self.files = pickle.load(f)
                
    def __getitem__(self, index):
        img_name = self.files[index]
        if self.image_path.find('birds')!=-1:
            clss = int(img_name.split('.')[0])
        else:
            clss = int(img_name[6:11])
        
        if  self.split == 'gen':   
            name,ext = os.path.splitext(img_name)
            if self.image_path.find('birds')!=-1:
                img_name = name + '.png'
            else:
                img_name = name + '.png'
            img_path = os.path.join(self.gen_root,img_name)
            img_path = img_path.replace('\\','/')
            img_path = img_path.replace('_128_','_256_')
            img = get_imgs(img_path)
            
        elif self.split == 'ground':
            name,ext = os.path.splitext(img_name)
            if self.image_path.find('birds')!=-1:
                img_name = name + '.npy'            
                img_path = os.path.join(self.ground_root,img_name)
                img_path = img_path.replace('\\','/')
                img = np.load(img_path)
            else:
                img_name = name + '.npy'            
                img_path = os.path.join(self.ground_root,img_name)
                img_path = img_path.replace('\\','/')
                img = np.load(img_path)
            

        return img, clss

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate the inception score')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data/102flowers/Oxford102/images_npy')    
    parser.add_argument('--gen_dir',dest='gen_dir',type=str,default='outputs/gan//flowers/TestImage')
    parser.add_argument('--exp_dir', dest='exp_dir', type=str, default='outputs/gan/flowers')   
    args = parser.parse_args()

    model = Inception_v3().cuda()
    model.eval()
    if args.data_dir.find('birds') != -1:
        dataset_ground = mAPData('evaluation/birds/filenames_ground_truth_image_CUB.pickle', 'ground',args.data_dir)
    else:
        dataset_ground = mAPData('evaluation/flowers/filenames_ground_truth_image_FLO.pickle', 'ground',args.data_dir)
    
    root_path = args.gen_dir
    epoch_names = os.listdir(root_path)
    save_path = os.path.join(args.exp_dir, 'mAP.text')
    info = 'starting evalute the mAP \n'
    with open(save_path, "a") as file:
        file.write(info)
    for epoch_name in epoch_names:
        epoch = int(epoch_name[5:])
        gen_dir = os.path.join(root_path,epoch_name)      
        if args.data_dir.find('birds') != -1:    
            dataset_gen = mAPData('evaluation/birds/filenames_Gen_image_CUB.pickle','gen',gen_dir)
        else:
            dataset_gen = mAPData('evaluation/flowers/filenames_Gen_image_FLO.pickle','gen',gen_dir)
        ground_loader = torch.utils.data.DataLoader(
            dataset_ground, batch_size=16,
            drop_last=False, shuffle=False)
        gen_loader = torch.utils.data.DataLoader(
            dataset_gen, batch_size=16,
            drop_last=False, shuffle=False)


        ground_imgs = []
        ground_cls = []
        for i, (img, clss) in enumerate(ground_loader):
            ground_imgs.append(img)
            ground_cls.append(clss)
        ground_imgs = torch.cat(ground_imgs,dim=0)
        ground_imgs = ground_imgs.squeeze(1)
        ground_cls = torch.cat(ground_cls)


        gen_imgs = []
        gen_cls = []
        for i, (img, cls) in enumerate(gen_loader):
            img = torch.tensor(img)
            img = img.float().cuda()
            feat = model(img)    
            gen_imgs.append(feat)
            gen_cls.append(cls)
        gen_imgs = torch.cat(gen_imgs,dim=0)
        gen_imgs = gen_imgs.squeeze(1)
        gen_cls = torch.cat(gen_cls)


        ground_f = normalizeFeature(ground_imgs)
        ground_f = ground_f.cuda()
        gen_f = normalizeFeature(gen_imgs)
        # cacualte mAP
        S = ground_f.mm(gen_f.t()) 
        value, indx = torch.sort(S,dim=1,descending=True)   
        sorted_cls = gen_cls[indx]   


        i = 0
        scores = 0
        R1=0
        R2=0
        R3=0
        R5=0
        R10=0
        total = 0
        for clss in ground_cls:
            sort = torch.where(sorted_cls[i]==clss)
            num = sort[0].shape[0]
            score = 0
            for j in range(num):
                score += (j+1.0)/(sort[0][j]+1.0).data.cpu().numpy()
                if j==0:
                    if sort[0][j]==0:
                        R1+=1
                if j<2:
                    if sort[0][j]<2:
                        R2+=1
                if j<3:
                    if sort[0][j]<3:
                        R3+=1
                if j<5:
                    if sort[0][j]<5:
                        R5+=1
                if j<10:
                    if sort[0][j]<10:
                        R10+=1
            score = score / num
            # print (score)
            scores += score
            

            i += 1
        mAP = scores / ground_cls.shape[0]
        R_1 = R1/ground_cls.shape[0]
        R_2 = R2/(2*ground_cls.shape[0])
        R_3 = R3/(3*ground_cls.shape[0])
        R_5 = R5/(5*ground_cls.shape[0])
        R_10 = R10/(10*ground_cls.shape[0])

        # caculate Recall

        info = ' Epoch: [{0}]  mAP: {mAP_:.4f} R@1: {R1_:.4f} R@2: {R2_:.4f} R@3: {R3_:.4f} R@5: {R5_:.4f} R@10: {R10_:.4f} \n'\
            .format(epoch,mAP_=mAP,R1_=R_1,R2_=R_2,R3_=R_3,R5_=R_5,R10_=R_10)        
        print(info)
        with open(save_path, "a") as file:
            file.write(info)
        







