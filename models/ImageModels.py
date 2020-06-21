import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models
from cfg.Pretrain.config import cfg


class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet34(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class VGG16(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(VGG16, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x


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

        # self.emb_features = conv1x1(768, self.nef)
        # self.emb_cnn_code = nn.Linear(2048, cfg.SPEECH.embedding_dim)

    # def init_trainable_weights(self):
    #     initrange = 0.1
    # #     self.emb_features.weight.data.uniform_(-initrange, initrange)
    #     self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

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
        return x #nn.functional.normalize(x, p=2, dim=1) #cnn_code  #1024



# in input of this network is the image feature
# extracted from the pre-trained model
class LINEAR_ENCODER(nn.Module):
    def __init__(self):
        super(LINEAR_ENCODER,self).__init__()
        self.L1 = nn.Linear(cfg.IMGF.input_dim,cfg.IMGF.embedding_dim)        
    
    def init_trainable_weights(self):
        initrange = 0.1
    #     self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.L1.weight.data.uniform_(-initrange, initrange)
    def forward(self, input):
        if len(input.shape)==3:
            input = input.squeeze(1)
        x = self.L1(input)
        return nn.functional.normalize(x,p=2,dim=1)

class LINEAR_ENCODER_2(nn.Module):
    def __init__(self):
        super(LINEAR_ENCODER_2,self).__init__()
        self.L1 = nn.Linear(cfg.IMGF.input_dim,cfg.IMGF.hid_dim)     
        self.L2 = nn.Linear(cfg.IMGF.hid_dim,cfg.IMGF.embedding_dim)
        self.b1 = nn.BatchNorm1d(cfg.IMGF.hid_dim)
        self.b2 = nn.BatchNorm1d(cfg.IMGF.embedding_dim)
        self.relu = nn.ReLU()
    def init_trainable_weights(self):
        initrange = 0.1
    #     self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.L1.weight.data.uniform_(-initrange, initrange)
        self.L2.weight.data.uniform_(-initrange, initrange)
    def forward(self, input):
        if len(input.shape)==3:
            input = input.squeeze(1)
        x = self.L1(input)
        x = self.b1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        return nn.functional.normalize(x,p=2,dim=1)


class LINEAR_DECODER(nn.Module):
    def __init__(self):
        super(LINEAR_DECODER,self).__init__()
        self.L1 = nn.Linear(cfg.IMGF.embedding_dim,cfg.IMGF.input_dim)    
    def init_trainable_weights(self):
        initrange = 0.1
    #     self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.L1.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, input):       
        x = self.L1(input)        
        return x



class LINEAR_DECODER_2(nn.Module):
    def __init__(self):
        super(LINEAR_DECODER,self).__init__()
        self.L1 = nn.Linear(cfg.IMGF.embedding_dim,cfg.IMGF.hid_dim)       
        self.L2 = nn.Linear(cfg.IMGF.hid_dim,cfg.IMGF.input_dim) 
        self.relu = nn.ReLU()
    
    def init_trainable_weights(self):
        initrange = 0.1
    #     self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.L1.weight.data.uniform_(-initrange, initrange)
        self.L2.weight.data.unifrom_(-initrange,initrange)
    def forward(self, input):
        if len(input.shape)==3:
            input = input.squeeze(1)
        x = self.L1(input)
        x = self.relu(x)
        x = self.L2(x)
        return x