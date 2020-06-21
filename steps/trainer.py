from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time

from torch.autograd import Variable
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

from utils.config import cfg
from utils.utils import mkdir_p
from models.model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024, INCEPTION_V3, MD_NET
from models.ImageModels import Inception_v3, LINEAR_ENCODER
from tensorboardX import FileWriter, summary



#Batch loss funcion

def batch_loss(cnn_code, rnn_code, class_ids,eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    batch_size = cfg.TRAIN.BATCH_SIZE
    labels = Variable(torch.LongTensor(range(batch_size)))
    labels = labels.cuda()   
    
    masks = []
    if class_ids is not None:
        class_ids =  class_ids.data.cpu().numpy()
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        masks = masks.to(torch.bool)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus,num_batches):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())
    if cfg.TREE.BRANCH_NUM > 3:
        netsD.append(D_NET512())
    if cfg.TREE.BRANCH_NUM > 4:
        netsD.append(D_NET1024())
    # TODO: if cfg.TREE.BRANCH_NUM > 5:
    # start_t = time.time()
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        # print(netsD[i])
    print('# of netsD', len(netsD))
    # end_t = time.time()
    # print('the duration is:%f'%(end_t-start_t) )
    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        epoch = cfg.TRAIN.NET_G[istart:iend]        
        count = int(epoch) * num_batches
        count = count + 1        

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)    

    inception_model = INCEPTION_V3()

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()

    if cfg.TRAIN.COEFF.MD_LOSS>0:  
        netMD = MD_NET()        
        netMD.apply(weights_init)
        netMD = torch.nn.DataParallel(netMD, device_ids=gpus)
    
    if cfg.TRAIN.NET_MD != '':
        state_dict = torch.load(cfg.TRAIN.NET_MD)
        netMD.load_state_dict(state_dict)
            
       
    if cfg.TRAIN.COEFF.MD_LOSS>0:
        return netG, netsD, netMD, len(netsD), inception_model, count
    else:
        return netG, netsD, len(netsD), inception_model, count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        optimizersD.append(opt)

    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, DIS_NET, epoch, model_dir):
    if cfg.TRAIN.COEFF.MD_LOSS>0:
        netsD, netMD = DIS_NET[0],DIS_NET[1]        
        torch.save(netMD.state_dict(), '%s/netMD.pth' % (model_dir))
    else:
        netsD = DIS_NET
    
    load_params(netG, avg_param_G)
    torch.save(netG.state_dict(), '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(netD.state_dict(), '%s/netD%d.pth' % (model_dir, i))
    # if cfg.TRAIN.COEFF.MD_LOSS>0:
    #     torch.save(netMD.state_dict(), '%s/netMD.pth' % (model_dir))
    print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs, epoch, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(real_img, '%s/epoch_%04dreal_samples.png' % (image_dir,epoch), normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    sup_real_img = summary.image('real_img', real_img_set, dataformats='HWC')
    summary_writer.add_summary(sup_real_img, epoch)

    for i in range(num_imgs):
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        vutils.save_image(fake_img.data, '%s/epoch_%04d_fake_samples%d.png' % (image_dir, epoch, i), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)

        sup_fake_img = summary.image('fake_img%d' % i, fake_img_set, dataformats='HWC')
        summary_writer.add_summary(sup_fake_img, epoch)
        summary_writer.flush()

# ################# Speech to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.testImage_dir = os.path.join(output_dir,'TestImage')
        if cfg.TRAIN.FLAG:            
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.testImage_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        imgs, w_imgs,s_imgs, t_embedding, class_id,_ = data

        real_vimgs, wrong_vimgs = [], []
        if cfg.CUDA:
            vembedding = Variable(t_embedding).cuda()
            same_vimg = Variable(s_imgs).cuda()
        else:
            vembedding = Variable(t_embedding)
            same_vimg = Variable(s_imgs)
        for i in range(self.num_Ds):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())                
            else:
                real_vimgs.append(Variable(imgs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
        
        return imgs, real_vimgs, wrong_vimgs,same_vimg, vembedding,class_id
    def train_MDnet(self, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        real_imgs = self.real_imgs[-1]
        wrong_imgs = self.wrong_imgs[-1]
        fake_imgs = self.fake_imgs[-1]
        similar_imgs = self.similar_imgs
        #        
        netMD = self.netMD
        optMD = self.optimizerMD
        netMD.zero_grad()
        
        same_labels = self.same_labels[:batch_size]        
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.real_labels[:batch_size]
        wrong_labels = self.wrong_labels[:batch_size]           

        
        real_feat = self.image_cnn(real_imgs.detach())
        real_feat = self.image_encoder(real_feat.detach())
        similar_feat = self.image_cnn(similar_imgs.detach())
        similar_feat = self.image_encoder(similar_feat.detach())
        fake_feat = self.image_cnn(fake_imgs.detach())
        fake_feat = self.image_encoder(fake_feat.detach())
        wrong_feat = self.image_cnn(wrong_imgs.detach())
        wrong_feat = self.image_encoder(wrong_feat.detach())

        same_logits = netMD(real_feat,real_feat)
        real_logits2 = netMD(real_feat,similar_feat)
        fake_logits2 = netMD(real_feat,fake_feat.detach())
        wrong_logits2 = netMD(real_feat,wrong_feat)
        
        errMD_si = cfg.TRAIN.COEFF.MD_LOSS*nn.CrossEntropyLoss()(real_logits2,real_labels.long())
        errMD_sa = cfg.TRAIN.COEFF.MD_LOSS*nn.CrossEntropyLoss()(same_logits,same_labels.long())
        errMD_fa = cfg.TRAIN.COEFF.MD_LOSS*nn.CrossEntropyLoss()(fake_logits2,fake_labels.long())
        errMD_wr = cfg.TRAIN.COEFF.MD_LOSS*nn.CrossEntropyLoss()(wrong_logits2,wrong_labels.long())
        if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
            errMD = errMD_si + errMD_sa + errMD_fa + errMD_wr    
        else:
            errMD = errMD_si + errMD_fa + errMD_wr   
    
        # backward
        errMD.backward()        
        optMD.step()
        # log
        if flag == 0:
            summary_MD = summary.scalar('MD_loss', errMD.item())
            self.summary_writer.add_summary(summary_MD, count)
        return errMD
    def train_Dnet(self, idx, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu = self.criterion, self.mu

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        #
        netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs.detach(), mu.detach())
        #
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(fake_logits[1], fake_labels)
            #
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            #
            errD = errD_real + errD_wrong + errD_fake
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD.backward()
        # update parameters
        optD.step()
        # log
        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, errD.item())
            self.summary_writer.add_summary(summary_D, count)
        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]      
       
        for i in range(self.num_Ds):
            outputs = self.netsD[i](self.fake_imgs[i], mu)
            errG = criterion(outputs[0], real_labels)
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(outputs[1], real_labels)
                errG = errG + errG_patch
            errG_total = errG_total + errG
            if cfg.TRAIN.COEFF.CONTENTCONSIST_LOSS>0 or cfg.TRAIN.COEFF.SEMANTICONSIST_LOSS>0 or cfg.TRAIN.COEFF.MD_LOSS>0: 
                fake_feat = self.image_cnn(self.fake_imgs[i])
                fake_feat = self.image_encoder(fake_feat)
            if cfg.TRAIN.COEFF.CONTENTCONSIST_LOSS>0 or cfg.TRAIN.COEFF.MD_LOSS>0:
                real_feat = self.image_cnn(self.real_imgs[i])
                real_feat = self.image_encoder(real_feat)
            if cfg.TRAIN.COEFF.CONTENTCONSIST_LOSS>0:                                
                loss1,loss2 = batch_loss(real_feat,fake_feat,self.class_ids)
                errG_CC = loss1 + loss2
                errG_total = errG_total + errG_CC*cfg.TRAIN.COEFF.CONTENTCONSIST_LOSS
            if cfg.TRAIN.COEFF.SEMANTICONSIST_LOSS>0:
                loss1,loss2 = batch_loss(self.txt_embedding,fake_feat,self.class_ids)
                errG_SC = loss1 + loss2
                errG_total = errG_total + errG_SC*cfg.TRAIN.COEFF.SEMANTICONSIST_LOSS
            if cfg.TRAIN.COEFF.MD_LOSS>0 and i==(self.num_Ds-1):             
                outputs2 = self.netMD(real_feat,fake_feat)
                errMG = nn.CrossEntropyLoss()(outputs2,real_labels.long())
                errG_total = errG_total + errMG*cfg.TRAIN.COEFF.MD_LOSS
            if flag == 0:
                summary_D = summary.scalar('G_loss%d' % i, errG.item())
                self.summary_writer.add_summary(summary_D, count)

        # Compute color consistency losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
                if flag == 0:
                    sum_mu = summary.scalar('G_like_mu2', like_mu2.item())
                    self.summary_writer.add_summary(sum_mu, global_step=count)

                    sum_cov = summary.scalar('G_like_cov2', like_cov2.item())
                    self.summary_writer.add_summary(sum_cov, global_step=count)
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1
                if flag == 0:
                    sum_mu = summary.scalar('G_like_mu1', like_mu1.item())
                    self.summary_writer.add_summary(sum_mu, count)

                    sum_cov = summary.scalar('G_like_cov1', like_cov1.item())
                    self.summary_writer.add_summary(sum_cov, count)

        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train(self):
        if cfg.TRAIN.COEFF.MD_LOSS>0:
            self.netG, self.netsD, self.netMD, self.num_Ds, self.inception_model, start_count = load_network(self.gpus,self.num_batches)
        else:        
            self.netG, self.netsD, self.num_Ds, self.inception_model, start_count = load_network(self.gpus,self.num_batches)
        
        
        avg_param_G = copy_G_params(self.netG)

        if cfg.TRAIN.COEFF.CONTENTCONSIST_LOSS>0 or cfg.TRAIN.COEFF.SEMANTICONSIST_LOSS>0 or cfg.TRAIN.COEFF.MD_LOSS>0:
            self.image_cnn = Inception_v3()
            self.image_encoder = LINEAR_ENCODER()
            if not isinstance(self.image_cnn, torch.nn.DataParallel):
                self.image_cnn = nn.DataParallel(self.image_cnn)
            if not isinstance(self.image_encoder, torch.nn.DataParallel):
                self.image_encoder = nn.DataParallel(self.image_encoder)
            if cfg.DATASET_NAME=='birds':
                self.image_encoder.load_state_dict(torch.load("outputs/pre_train/birds/models/best_image_model.pth"))
            if cfg.DATASET_NAME=='flowers':
                self.image_encoder.load_state_dict(torch.load("outputs/pre_train/flowers/models/best_image_model.pth"))
            
            if cfg.CUDA:
                self.image_cnn = self.image_cnn.cuda()
                self.image_encoder = self.image_encoder.cuda()
            self.image_cnn.eval()
            self.image_encoder.eval()
            for p in self.image_cnn.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.optimizerG, self.optimizersD = define_optimizers(self.netG, self.netsD)
        if cfg.TRAIN.COEFF.MD_LOSS>0:        
            self.optimizerMD = optim.Adam(self.netMD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))
        self.same_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))
        self.wrong_labels = Variable(torch.FloatTensor(self.batch_size).fill_(2))

        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.same_labels = self.same_labels.cuda()
            self.wrong_labels = self.wrong_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)       
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, self.similar_imgs,self.txt_embedding, self.class_ids = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                self.fake_imgs, self.mu, self.logvar = self.netG(noise, self.txt_embedding)

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD
                #update MD network
                errMD = self.train_MDnet(count)
                errD_total += errMD
                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                kl_loss, errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)      # 

                # for inception score
                # pred = self.inception_model(self.fake_imgs[-1].detach())
                # predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.item())
                    summary_G = summary.scalar('G_loss', errG_total.item())
                    summary_KL = summary.scalar('KL_loss', kl_loss.item())
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                count = count + 1
            if epoch % cfg.TRAIN.SAVE_EPOCH ==0:
                if cfg.TRAIN.COEFF.MD_LOSS>0:
                    DIS_NET = [self.netsD,self.netMD]
                else: 
                    DIS_NET = self.netsD
                save_model(self.netG, avg_param_G, DIS_NET, epoch, self.model_dir)
            if epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0:                                               
                # Save images
                backup_para = copy_G_params(self.netG)
                load_params(self.netG, avg_param_G)
                #
                self.fake_imgs, _, _ = self.netG(fixed_noise, self.txt_embedding)
                save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds, epoch, self.image_dir,
                                    self.summary_writer)
                #
                load_params(self.netG, backup_para)

                #############################
                #***during the training process, the paramerter of G are updated alone
                #**why in the generating stage, use the weighting parameter of G

                #############################
                """
                # Compute inception score
                if len(predictions) > 500:
                    predictions = np.concatenate(predictions, 0)
                    mean, std = compute_inception_score(predictions, 10)
                    # print('mean:', mean, 'std', std)
                    m_incep = summary.scalar('Inception_mean', mean)
                    self.summary_writer.add_summary(m_incep, count)
                    #
                    mean_nlpp, std_nlpp = negative_log_posterior_probability(predictions, 10)
                    m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                    self.summary_writer.add_summary(m_nlpp, count)
                    #
                    predictions = []
                """

            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     kl_loss.item(), end_t - start_t))

        if cfg.TRAIN.COEFF.MD_LOSS>0:
            DIS_NET = [self.netsD,self.netMD]
        else: 
            DIS_NET = self.netsD
        save_model(self.netG, avg_param_G, DIS_NET, epoch, self.model_dir)
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames, save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' % (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames, save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/%s' % (save_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        NET_G_root = self.model_dir
        net_list = os.listdir(NET_G_root)
        G_NETS = []
        for net in net_list:
            if net.find('netG') != -1:
                s_tmp = net
                istart = s_tmp.rfind('_') + 1
                iend = s_tmp.rfind('.')
                epoch = int(s_tmp[istart:iend])
                if epoch>=100 and epoch<=600:        ##################********************************************250  
                    G_NETS.append(net)
        
        for NET_G in G_NETS:
            NET_G_path = os.path.join(NET_G_root,NET_G)
            if split_dir == 'test':
                split_dir = 'valid'
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = torch.load(NET_G_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', NET_G_path)

            # the path to save generated images
            s_tmp = NET_G_path
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            epoch = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/epoch%d' % (self.testImage_dir, epoch)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)

                fake_img_list = []
                for i in range(embedding_dim):
                    fake_imgs, _, _ = netG(noise,t_embeddings[:, i, :])   #t_embeddings[:, i, :] by shawn
                    if cfg.TEST.B_EXAMPLE:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                        fake_img_list.append(fake_imgs[2].data.cpu())
                    else:
                        self.save_singleimages(fake_imgs[-1], filenames, save_dir, split_dir, i, 256)
                        # self.save_singleimages(fake_imgs[-2], filenames,
                        #                        save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                    # break
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames, save_dir, split_dir, 256)
