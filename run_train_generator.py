
import argparse

from src.main.generator import TrainGenerator
import os 
#from utils.utils import str2bool


TRAIN_TYPE = ["pretrain","gen","end2end"]

parser = argparse.ArgumentParser(description='Code for training a lip sync generator via landmark')
""" ---------- Dataset --------"""
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/home/peter/Peter/audio-visual/dataset/lrs2_main_fl_256_full_face_prepro/', type=str)

""" --------- Generator --------"""
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='./checkpoints/generator/', type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoints', default=None, type=str)

"""---------- SyncNet ----------"""

parser.add_argument('--train_type',
                    help='--train_type select "pretrain" for training generator with pretrain SyncNet, "gen" for training only generator without SyncNet, and "end2end" for training generator and SyncNet together', 
                    default="pretrain", type=str)
parser.add_argument('--pretrain_syncnet_path', help="Path of pretrain syncnet", default='./checkpoints/syncnet/pretrain_cen_face_syncnet_16.pth')

"""---------- Save name --------"""

parser.add_argument("--checkpoint_interval", help="Checkpoint interval and eval video", default=10, type=int)
parser.add_argument('--save_name', help='name of a save', default="pretrain030_b32_mse_loss", type=str)


args = parser.parse_args()


def main():

    if (args.train_type  not in TRAIN_TYPE):

        raise ValueError("Argument --train_type mus be in {}".format(TRAIN_TYPE))

    if not os.path.exists(args.data_root):
        raise ValueError("Data root  does not exist")

    if args.checkpoint_path is not None and not os.path.exists(args.checkpoint_path):

        raise ValueError("Checkpoint for  Generator does not exists")

    if args.save_name is None:

        raise ValueError('Please provide a save name')

    if args.checkpoint_dir is None:

        raise ValueError("Please provide a checkpoint_dir")

    if args.train_type == "pretrain" and not os.path.exists(args.pretrain_syncnet_path):

        raise ValueError("Please provide a checkpoint_path for pretrain_syncnet for using pretrain discriminator")

     # if create checkpoint dir if it does not exist
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    model = TrainGenerator(args=args)

    model.start_training()


if __name__ == "__main__":

    main()
