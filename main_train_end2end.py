
import argparse
from src.main.end2end import End2End
import os 
from utils.utils import str2bool



parser = argparse.ArgumentParser(description='Code for training a lip sync generator via landmark')
""" ---------- Dataset --------"""
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/home/peter/Peter/audio-visual/dataset/lrs2_main_fl_256_full_face_prepro/', type=str)

""" --------- Generator --------"""
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='./checkpoints/generator/', type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoints', default=None, type=str)

"""---------- SyncNet ----------"""
# path of pretrain syncnet weight
parser.add_argument('--checkpoint_syncnet_path', help="Checkpoint for pretrained SyncNet", default='./src/models/ckpt/1361_sync.pth' ,type=str)
parser.add_argument('--apply_disc',help="Apply discriminator at epoch ", default=10)

"""---------- Save name --------"""
parser.add_argument('--save_name', help='name of a save', default="TO_showing_inmeeting_gen_end2end_with_disc_at10", type=str)


args = parser.parse_args()


def main():
     # if create checkpoint dir if it does not exist
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    model = End2End(args=args)

    model.start_training()


if __name__ == "__main__":

    main()
