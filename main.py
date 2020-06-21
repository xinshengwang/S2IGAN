from __future__ import print_function
from utils.config import cfg, cfg_from_file
from dataset.datasets import SpeechDataset

import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import sys
import numpy as np

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/eval_places.yml', type=str)    # train: birds_3stages.yml  test: eval_birds.yml
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data/places/7classes')
    parser.add_argument('--save_root',type=str,default='outputs/gan/places',help='The root path for both pre-train result and main results')
    parser.add_argument('--manualSeed', type=int,default=200,help='manual seed')
    parser.add_argument('--WORKERS',type=int,default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.save_root != '':
        cfg.save_root = args.save_root
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
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
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = cfg.save_root

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:        
        bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    
    dataset = SpeechDataset(cfg.DATA_DIR, split_dir,cfg.EMBEDDING_TYPE,
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
                                             drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS),worker_init_fn=worker_init_fn)

    # Define models and go to train/evaluate
    if not cfg.GAN.B_CONDITION:
        from steps.trainer import GANTrainer as trainer
    else:
        from steps.trainer import condGANTrainer as trainer
    algo = trainer(output_dir, dataloader, imsize)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
