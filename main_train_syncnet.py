
import argparse
import warnings
from src.main.syncnet import TrainSync
import os 
from utils.utils import str2bool
import torch
warnings.simplefilter(action='ignore', category=FutureWarning)

    
parser = argparse.ArgumentParser(description="Code for training SyncNet with the lip key points ")

parser.add_argument("--data_root", help="Root for preprocessed lip keypoint dataset", default='/home/peter/Peter/audio-visual/dataset/lrs2_main_fl_256_full_face_prepro')

parser.add_argument("--checkpoint_dir", help="dir to save checkpoints for SyncNet for lip keypoint", default='./checkpoints/syncnet/',
                    type=str)

parser.add_argument('--checkpoint_path', help="Resume from checkpoints or testing a model from checkpoints", default=None, type=str)

parser.add_argument('--save_name', help="name of a save", default="syncnet_test_pretrain",type=str)

parser.add_argument('--do_train' , help="Train a mode or testing a model", default='True' , type=str2bool)


args = parser.parse_args()



model = TrainSync(args=args)


if args.do_train :
    
    model.start_training()
    
else:
    
    if args.checkpoint_path is None:
        
        raise ValueError("Required the path of model's checkpoint for Testing model --checkpoint_path")
    
    if not os.path.exists(args.checkpoint_path):
        
        raise ValueError("Give path for model checkpoint does not exists")
    
    model.start_testing()
        
        
    
    
