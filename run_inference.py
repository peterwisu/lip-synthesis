# Inference model



import argparse

from src.main.inference import Inference
import os


MODEL_TYPE = ['lstm','attn_lstm']

parser = argparse.ArgumentParser(description="File for running Inference")

parser.add_argument('--model_type', help='Type of generator model', default='attn_lstm', type=str)

parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default='./checkpoints/generator/attn_lstmgen_syncloss.pth')

parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default='./checkpoints/image2image/ckpt_116_i2i_comb.pth',required=False)

parser.add_argument('--input_face', type=str, help="File path for input videos/images contain face",default='./dummy/me.png', required=False)

parser.add_argument('--input_audio', type=str, help="File path for input audio/speech as .wav files", default='./dummy/test_audio.mp3', required=False)

# parser.add_argument('--output_path', type=str, help="Path for saving the result", default='result.mp4', required=False)

parser.add_argument('--fps', type=float, help= "Can only be specified only if using static image default(25 FPS)", default=25,required=False)

parser.add_argument('--fl_detector_batchsize',  type=int , help='Batch size for landmark detection', default = 64)

parser.add_argument('--generator_batchsize', type=int, help="Batch size for Generator model", default=2) 

parser.add_argument('--seq_len', type=int, help="Sequence length for Generator model", default=5) 

parser.add_argument('--output_name', type=str , help="Name and path of the output file", default="results.mp4")

parser.add_argument('--vis_fl', type=bool, help="Visualize Facial Landmark ??", default=False)

parser.add_argument('--only_fl', type=bool, help="Visualize only Facial Landmark ??", default=False)

parser.add_argument('--test_img2img', type=bool, help="Testing image2image module with no lip generation" , default=False)




args = parser.parse_args()


def main(args):

 
    if (args.model_type  not in MODEL_TYPE):

        raise ValueError("Argument --model_type mus be in {}".format(MODEL_TYPE))   

    import time 



    start_time = time.time()

    inference = Inference(args=args)

    inference.start()

    end_time =  time.time()

    duration = end_time - start_time


    print("Time Taken {}".format(duration))


if __name__ == "__main__":
    
    main(args=args)
