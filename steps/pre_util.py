import math
import pickle
import numpy as np
import torch
from cfg.Pretrain.config import cfg
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def calc_mAP(image_output,audio_output,cls_id):

    value,idx = cls_id.sort()
    image_output = image_output[idx]
    audio_output = audio_output[idx]
    cls_id = cls_id[idx]
    cls_f = -1
    new_cls = []      # classes of the sampled audio
    cls_num = []      #number of each classes of sampled audio
    sampled_audio = []
    i = 0
    j = 0
   

    # with this code, the query is speech
    # only one speech for one class images
    for cls_i in cls_id:
        if cls_i!= cls_f:
            new_cls.append(cls_i.unsqueeze(0))  
            sampled_audio.append(audio_output[i].unsqueeze(0))         
            cls_f = cls_i            
            if i!=0:
                cls_num.append(j)    #             
            j = 1 
        else:
            j += 1
        i += 1   
    #the query is speech
    #the query is the average of speeches for each class images 
    """
    for cls_i in cls_id:
        if cls_i!= cls_f:
            new_cls.append(cls_i.unsqueeze(0))  
            sampled_audio.append(audio_output[i].unsqueeze(0))         
            cls_f = cls_i            
            if i!=0:
                cls_num.append(j)    #             
            j = 1 
        else:
            j += 1
        i += 1
    """
    cls_num.append(j)   


    new_cls = torch.cat(new_cls)
    sampled_audio = torch.cat(sampled_audio)
       
    # using consine similarity
    if cfg.EVALUATE.dist == 'cosine':
        img_f = normalizeFeature(image_output)
        aud_f = normalizeFeature(sampled_audio) 
        S = aud_f.mm(img_f.t()) 
        value, indx = torch.sort(S,dim=1,descending=True)
    elif cf.EVALUATE.dist == 'L2':
        img_f = image_output / image_output.norm(dim=1,keepdim=True)
        aud_f = sampled_audio / sampled_audio.norm(dim=1,keepdim=True)
        img_ex = (img_f.unsqueeze(0)).repeat(aud_f.shape[0],1,1)
        aud_ex = (aud_f.unsqueeze(1)).repeat(1,img_f.shape[0],1)
        diff = aud_ex - img_ex
        squareDiff = diff**2
        squareDist = squareDiff.sum(-1)
        S = squareDist**0.5
        value, indx = torch.sort(S,dim=1)
    
    
    class_sorted = cls_id[indx]
    clss_m2 = new_cls.unsqueeze(-1).repeat(1,S.shape[1])
    
    mask = (class_sorted==clss_m2).bool()
    class_sorted_filed = class_sorted.data.masked_fill_(mask,-10e5)   

    v, index = torch.sort(class_sorted_filed,dim=1)
    index = index +1
    sc = 0.0
    ap = 0.0


    for i in range(index.shape[0]):
        sc = 0.0
        num = cls_num[i]
        for k in range(num):    
            position =  index[i][:num]  
            position = sorted(position)     
            sc += (k+1.0)/(position[k]).float()
        ap += sc/cls_num[i]
    
    mAP = ap/(mask.shape[0])

    return mAP






def calc_recalls(image_outputs, audio_outputs):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    # S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    image_L2  = image_outputs.pow(2).sum(1).pow(0.5).unsqueeze(-1)
    audio_L2  = audio_outputs.pow(2).sum(1).pow(0.5).unsqueeze(-1)
    S = (image_outputs.mm(audio_outputs.t()))/(image_L2.mm(audio_L2.t()))

    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def retrieval_evaluation_all(image_output,audio_output,I_id,A_id):      
    # using consine similarity    
    img_f = normalizeFeature(image_output)
    aud_f = normalizeFeature(audio_output) 
    S = img_f.mm(aud_f.t())
    #for image to audio retrieval
    _, indx_I2A = torch.sort(S,dim=1,descending=True)
    class_sorted_I2A = A_id[indx_I2A]
    Correct_num_I2A_1 = sum(class_sorted_I2A[:,0]==I_id)
    Correct_num_I2A_5 = ((class_sorted_I2A[:,:5]==I_id.unsqueeze(-1).repeat(1,5)).sum(1) != 0 ).sum()
    Correct_num_I2A_10 = ((class_sorted_I2A[:,:10]==I_id.unsqueeze(-1).repeat(1,10)).sum(1) != 0 ).sum()
    
    Rank1_I2A = Correct_num_I2A_1*1.0/img_f.shape[0]
    Rank5_I2A = Correct_num_I2A_5*1.0/img_f.shape[0]
    Rank10_I2A = Correct_num_I2A_10*1.0/img_f.shape[0]
    
    Rank_I2A = torch.nonzero(class_sorted_I2A==I_id.unsqueeze(-1))
    kr,inverse = torch.unique(Rank_I2A[:,0],sorted = False,return_inverse = True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(kr.size(0)).scatter_(0, inverse, perm)
    
    medr_I2A = Rank_I2A[perm][:,1].median()
    
    
    
    # mAP_I2A = AP_K(indx_I2A,I_id,A_id)
    # mAP_I2A = 0.0
    

    # for audio to image retrieval
    S_T = S.T
    _, indx_A2I = torch.sort(S_T,dim=1,descending=True)
    class_sorted_A2I = I_id[indx_A2I]
    Correct_num_A2I_1 = sum(class_sorted_A2I[:,0]==A_id)
    Correct_num_A2I_5 = ((class_sorted_A2I[:,:5]==A_id.unsqueeze(-1).repeat(1,5)).sum(1) != 0 ).sum()
    Correct_num_A2I_10 = ((class_sorted_A2I[:,:10]==A_id.unsqueeze(-1).repeat(1,10)).sum(1) != 0 ).sum()


    Rank1_A2I_1 = Correct_num_A2I_1*1.0/aud_f.shape[0]
    Rank1_A2I_5 = Correct_num_A2I_5*1.0/aud_f.shape[0]
    Rank1_A2I_10 = Correct_num_A2I_10*1.0/aud_f.shape[0]
   
    Rank_A2I = torch.nonzero(class_sorted_A2I==A_id.unsqueeze(-1))
    kr,inverse = torch.unique(Rank_A2I[:,0],sorted = False,return_inverse = True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(kr.size(0)).scatter_(0, inverse, perm)    
    medr_A2I = Rank_A2I[perm][:,1].median()

    # mAP_A2I = AP_K(indx_A2I,A_id,I_id)
    # mAP_A2I = 0.0
    recalls = {'A_r1':Rank1_A2I_1, 'A_r5':Rank1_A2I_5, 'A_r10':Rank1_A2I_10,
                'I_r1':Rank1_I2A, 'I_r5':Rank5_I2A, 'I_r10':Rank10_I2A,
                'medr_I2A':medr_I2A, 'medr_A2I':medr_A2I}
    return recalls



def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def normalizeFeature(x):	
    
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat

def triplet_loss(image_output, audio_output, neg_samples):

    p_dist = 1 - torch.cosine_similarity(image_output,audio_output,dim=1)
    n_dist = 1 - torch.cosine_similarity(image_output,neg_samples,dim=1)

    loss = (cfg.margin + p_dist - n_dist).sum()/(p_dist.shape[0])    
    loss = torch.max(torch.tensor(0).float().cuda(),loss)  

    return loss

# top k% negative mining
def negative_samples_mining(image_output,audio_output,cls_id):
    img_f = normalizeFeature(image_output)
    audo_f = normalizeFeature(audio_output)
    sim_mat = img_f.mm(audo_f.t())

    clss_m1 = cls_id.unsqueeze(-1).repeat(1,cls_id.shape[0])
    clss_m2 = clss_m1.t()

    mask = (clss_m1==clss_m2).bool().cuda()
    n_mask = (clss_m1!=clss_m2)
    sim_mat.data.masked_fill_(mask,-1.0)

    # sim_mat = sim_mat*mask
    sim, index = sim_mat.sort(1,descending=True)
    statistic = n_mask.int()
    num = statistic.sum(1)
    min_num = num.min()
    number = (min_num.float()*0.1).int()
    idx_i = np.random.randint(0,number,size=(image_output.shape[0]))
 
    for i in range(index.shape[0]):
        idx = index[i][idx_i[i]].cpu().numpy()
        if i==0:
            idxes = idx
        else:
            idxes = np.hstack((idxes,idx))   
    
    
    neg_samples = audio_output[idxes]
    return neg_samples

# batch hardest negative mining
def hardest_negative_mining_pair(image_output,audio_output,cls_id):
    img_f = normalizeFeature(image_output)
    audo_f = normalizeFeature(audio_output)
    sim_mat = img_f.mm(audo_f.t())

    clss_m1 = cls_id.unsqueeze(-1).repeat(1,cls_id.shape[0])
    clss_m2 = clss_m1.t()

    mask = (clss_m1==clss_m2).bool().cuda()
    n_mask = (clss_m1!=clss_m2)
    sim_mat.data.masked_fill_(mask,-1.0)
    sim_mat_T = sim_mat.t()
    # sim_mat = sim_mat*mask
    sim, index = sim_mat.sort(1,descending=True)
    sim_t,index_t = sim_mat_T.sort(1,descending=True)
 
    for i in range(index.shape[0]):
        idx = index[i][0].cpu().numpy()
        idxt = index_t[i][0].cpu().numpy()
        if i==0:
            idxes = idx
            idxes_t = idxt
        else:
            idxes = np.hstack((idxes,idx))   
            idxes_t = np.hstack((idxes_t,idxt))
    
    
    neg_audio = audio_output[idxes]
    neg_img = image_output[idxes_t]
    return neg_audio, neg_img

def hardest_negative_mining_single(image_output,cls_id):
    img_f = normalizeFeature(image_output)
    audo_f = img_f
    sim_mat = img_f.mm(audo_f.t())

    clss_m1 = cls_id.unsqueeze(-1).repeat(1,cls_id.shape[0])
    clss_m2 = clss_m1.t()

    mask = (clss_m1==clss_m2).bool().cuda()
    n_mask = (clss_m1!=clss_m2)
    sim_mat.data.masked_fill_(mask,-1.0)   
    # sim_mat = sim_mat*mask
    sim, index = sim_mat.sort(1,descending=True)   
 
    for i in range(index.shape[0]):
        idx = index[i][0].cpu().numpy()
        if i==0:
            idxes = idx
        else:
            idxes = np.hstack((idxes,idx))               
    
    
    neg_samples = image_output[idxes]   
    return neg_samples



# batch loss

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


def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def distribute_loss(img,audio):
    soft_image = F.softmax(img)
    soft_audio = F.softmax(audio)
    log_soft_image = F.log_softmax(img)
    log_soft_audio = F.log_softmax(audio)
    loss1 = soft_image.mul(log_soft_audio).sum(1).mean()*(-1.0)
    loss2 = soft_audio.mul(log_soft_image).sum(1).mean()*(-1.0)
    return loss1, loss2


def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10
