
import argparse
import warnings
from src.main.syncnet import TrainSyncNet
import os 
from utils.utils import str2bool
from datetime import date 
warnings.simplefilter(action='ignore', category=FutureWarning)

    
parser = argparse.ArgumentParser(description="Code for training SyncNet with the lip key points ")

parser.add_argument("--data_root", help="Root for preprocessed lip keypoint dataset", default='/home/peter/Peter/audio-visual/dataset/lrs2_main_fl_256_full_face_prepro')

parser.add_argument("--checkpoint_dir", help="dir to save checkpoints for SyncNet for lip keypoint", default='./checkpoints/syncnet/',
                    type=str)

parser.add_argument('--checkpoint_path', help="Resume from checkpoints or testing a model from checkpoints", default=None)

parser.add_argument('--save_name', help="name of a save", default="test",type=str)

parser.add_argument('--do_train' , help="Train a mode or testing a model", default='True' , type=str2bool)


args = parser.parse_args()

def main():

    if args.do_train :

        if not os.path.exists(args.data_root):
            raise ValueError("Data root  does not exist")


        if args.checkpoint_path is not None and not os.path.exists(args.checkpoint_path):

            raise ValueError("Checkpoint for SyncNet does not exists")

       
        if args.checkpoint_dir is None:

            raise ValueError("Please provide a checkpoint_dir")

         # if create checkpoint dir if it does not exist
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)


        model = TrainSyncNet(args=args)
        model.start_training()

    else : 
 
        if args.checkpoint_path is None:
            
            raise ValueError("Required the path of model's checkpoint for Testing model --checkpoint_path")
        
        if not os.path.exists(args.checkpoint_path):
            
            raise ValueError("Give path for model checkpoint does not exists")


        model = TrainSyncNet(args=args)
        model.start_testing()

    

if __name__  == "__main__":

    main()
        
        
    
