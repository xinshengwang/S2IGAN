import argparse
import os
import pickle
import sys
import time
import torch
import random
import datetime
import pprint
import dateutil.tz
import numpy as np
from PIL import Image
import sys
sys.path.append("..")

from dataset.datasets_pre import SpeechDataset, pad_collate 
from models import AudioModels, ImageModels, classification

from steps.pre_traintest import train, validate, feat_extract_co
import torchvision.transforms as transforms 
from cfg.Pretrain.config import cfg, cfg_from_file

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', type = str, default='data/places/7classes') #
parser.add_argument('--exp_dir', type = str, default= '')
parser.add_argument('--save_root', type=str, default='outputs/pre_train/places')
parser.add_argument("--resume", action="store_true", default=True,
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="adam",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=24, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate-speech', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=50, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
    metavar='W', help='weight decay (default: 1e-4)')     #5e-7
parser.add_argument("--n_epochs", type=int, default=120,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=2,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet",
        help="audio model architecture", choices=["Davenet"])
parser.add_argument("--image-model", type=str, default="VGG16",
        help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", action="store_true",
    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
parser.add_argument('--tasks',type = str, default='extraction', help="training | extraction")

parser.add_argument('--rnn-type',type = str, default='GRU', help='LSTM | GRU')
parser.add_argument('--cfg_file',type = str, default='cfg/Pretrain/places_eval.yml',help='optional config file')
parser.add_argument('--img_size',type = int, default=256, help = 'the size of image')

parser.add_argument('--gpu_id',type = int, default= 0)
parser.add_argument('--manualSeed',type=int,default= 200, help='manual seed')

args = parser.parse_args()

resume = args.resume

print(args)

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

if args.gpu_id != -1:
    cfg.GPU_ID = args.gpu_id
else:
    cfg.CUDA = False

if args.data_path != '':
    cfg.DATA_DIR = args.data_path
# print('Using config:')
# pprint.pprint(cfg)
if args.batch_size != None:
    cfg.TRAIN.BATCH_SIZE = args.batch_size



cfg.exp_dir = os.path.join(args.save_root,'pre-train')

if not cfg.TRAIN.FLAG:
    args.manualSeed = 200
elif args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if cfg.CUDA:    
    torch.cuda.manual_seed(args.manualSeed)  
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(args.manualSeed + worker_id)


now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = '../output/%s_%s_%s' % \
    (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

split_dir, bshuffle = 'train', True
if not cfg.TRAIN.FLAG:
    # bshuffle = False
    split_dir = 'test'

# Get data loader
imsize = args.img_size
image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

if cfg.TRAIN.MODAL == 'co-train':
    dataset = SpeechDataset(cfg.DATA_DIR, 'train',
                            img_size = imsize,
                            transform=image_transform)
    dataset_test = SpeechDataset(cfg.DATA_DIR, 'test',
                            img_size = imsize,
                            transform=image_transform)


    assert dataset


    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle,num_workers=cfg.WORKERS,collate_fn=pad_collate,worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False,num_workers=cfg.WORKERS,collate_fn=pad_collate,worker_init_fn=worker_init_fn)


# Dataloader for classificaiton of single modal
elif cfg.TRAIN.MODAL == 'extraction':
    
    dataset = SpeechDataset(cfg.DATA_DIR, 'train',
                            img_size = imsize,
                            transform=image_transform)
    dataset_test = SpeechDataset(cfg.DATA_DIR, 'test',
                            img_size = imsize,
                            transform=image_transform)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False,num_workers=cfg.WORKERS,collate_fn=pad_collate,worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False,num_workers=cfg.WORKERS,collate_fn=pad_collate,worker_init_fn=worker_init_fn)

if cfg.Loss.deco:
    decoder = ImageModels.LINEAR_DECODER()
if cfg.DATA_DIR.find('birds') != -1 or cfg.DATA_DIR.find('flowers') != -1:
    class_model = classification.CLASSIFIER()


if cfg.SPEECH.model == 'RNN':
    audio_model = AudioModels.RNN_ENCODER(cfg.SPEECH.input_dim, cfg.SPEECH.hidden_size,cfg.SPEECH.num_layers)
elif cfg.SPEECH.model == 'CRNN':
    audio_model = AudioModels.CNN_RNN_ENCODER()
elif cfg.SPEECH.model == 'CNN':
    audio_model = AudioModels.CNN_ENCODER(cfg.SPEECH.embedding_dim)

image_cnn = ImageModels.Inception_v3()
image_model = ImageModels.LINEAR_ENCODER()


# train(audio_model, image_model,train_loader, val_loader, args)

if cfg.TRAIN.MODAL == 'co-train':
    if cfg.DATA_DIR.find('birds') != -1 or cfg.DATA_DIR.find('flowers') != -1:
        MODELS = [audio_model, image_cnn,image_model,class_model]
    else:
        MODELS = [audio_model, image_cnn,image_model]
        
    train(MODELS,train_loader, val_loader, args)
    
if cfg.TRAIN.MODAL == 'extraction':
    feat_extract_co(audio_model,cfg.DATA_DIR,args)
