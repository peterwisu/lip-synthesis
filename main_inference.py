

import argparse
from src.main.inference import Inference
import os

parser = argparse.ArgumentParser(description="File for running Inference")

parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default='./checkpoints/generator/demo_front.pth')

parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default='./checkpoints/image2image/image2image.pth',required=False)

parser.add_argument('--input_face', type=str, help="File path for input videos/images contain face",default='dummy/00001.mp4', required=False)

parser.add_argument('--input_audio', type=str, help="File path for input audio/speech as .wav files", default='./dummy/audio.mp3', required=False)

# parser.add_argument('--output_path', type=str, help="Path for saving the result", default='result.mp4', required=False)

parser.add_argument('--fps', type=float, help= "Can only be specified only if using static image default(25 FPS)", default=25,required=False)

parser.add_argument('--fl_detector_batchsize',  type=int , help='Batch size for landmark detection', default = 32)

parser.add_argument('--generator_batchsize', type=int, help="Batch size for Generator model", default=16)

parser.add_argument('--output_name', type=str , help="Name and path of the output file", default="results.mp4")

parser.add_argument('--vis_fl', type=bool, help="Visualize Facial Landmark ??", default=True)

args = parser.parse_args()



def main(args):

    

    inference = Inference(args=args)

    inference.start()


if __name__ == "__main__":
    
    main(args=args)
