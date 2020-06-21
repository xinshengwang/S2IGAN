from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C


__C.cmvn = False
__C.margin = 0.5
# __C.save_root = 'outputs/test'
# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.DATASET_ALL_CLSS_NUM = 200 #200
__C.DATASET_TRAIN_CLSS_NUM =150 #150
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 8 # 8
__C.result_file = 'baseline.text'

__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = False


__C.add_noise = False
__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 256


# Training options
__C.TRAIN = edict()
__C.TRAIN.MODAL = 'co-train'     # co-train | classification | teacher_student | extraction
__C.TRAIN.BATCH_SIZE = 128    #128
__C.TRAIN.MAX_EPOCH = 250
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

__C.EXTRACT = edict()
__C.EXTRACT.split = 'train'

__C.CROSS_ATT = edict()
__C.CROSS_ATT.att = False
__C.CROSS_ATT.act = 'sigmoid'   # softmax | sigmoid
__C.CROSS_ATT.smooth_soft = 1.0
__C.CROSS_ATT.smooth_sigm = 0.1

               


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


__C.IMG = edict()
__C.IMG.style = 'pickle'   #npy | raw | pickle

__C.IMGF = edict()
__C.IMGF.Layer = 1
__C.IMGF.input_dim = 2048
__C.IMGF.hid_dim = 1600
__C.IMGF.embedding_dim = 1024




__C.rnn_type = 'LSTM'

__C.SPEECH = edict()
__C.SPEECH.style= 'mel'   #npy | WAV
__C.SPEECH.model = 'CRNN'   #CNN | CRNN | RNN
__C.SPEECH.self_att = True
__C.SPEECH.CAPTIONS_PER_IMAGE = 10
__C.SPEECH.window_size = 25
__C.SPEECH.stride = 10
__C.SPEECH.input_dim = 40
__C.SPEECH.hidden_size = 512
__C.SPEECH.embedding_dim = 1024
__C.SPEECH.num_layers = 2
__C.SPEECH.sample = 22050
 

__C.CNNRNN = edict()
__C.CNNRNN.rnn_type = 'GRU'
__C.CNNRNN.in_channels = 40    #40
__C.CNNRNN.hid_channels = 50
__C.CNNRNN.hid2_channels = 64
__C.CNNRNN.out_channels = 128  #64
__C.CNNRNN.kernel_size = 6
__C.CNNRNN.stride = 2
__C.CNNRNN.padding = 0

__C.CNNRNN_RNN = edict()
__C.CNNRNN_RNN.input_size = 128     #64    
__C.CNNRNN_RNN.hidden_size = 512
__C.CNNRNN_RNN.num_layers = 2
__C.CNNRNN_RNN.dropout = 0.0
__C.CNNRNN_RNN.bidirectional = True

__C.CNNRNN_ATT = edict()
__C.CNNRNN_ATT.in_size = 1024
__C.CNNRNN_ATT.hidden_size = 128
__C.CNNRNN_ATT.n_heads = 1


__C.CLASSIFICATION = edict()
__C.CLASSIFICATION.data = 'audio'

__C.EVALUATE = edict()
__C.EVALUATE.dist = 'cosine'   # consine | L2



__C.SPEECH.cmvn = True  # apply CMVN on feature


__C.Loss  = edict() 
__C.Loss.clss = True
__C.Loss.cont = False
__C.Loss.hinge = False
__C.Loss.batch = True
__C.Loss.KL = False
__C.Loss.deco = False
__C.Loss.adv = False
__C.Loss.trip = False
__C.Loss.dist = False
__C.Loss.gamma_clss = 1.0
__C.Loss.gamma_cont = 1.0
__C.Loss.gamma_hinge = 1.0
__C.Loss.gamma_batch = 1.0
__C.Loss.gamma_KL = 1.0
__C.Loss.gamma_deco = 1.0
__C.Loss.gamma_adv = 1.0
__C.Loss.hinge_margin = 1.0
__C.Loss.gamma_trip = 1.0
__C.Loss.trip_margin = 1.0
__C.Loss.gamma_dist = 1.0
__C.Loss.dist_T = 1.0
__C.Loss.adv_k = 5

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
