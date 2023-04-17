import torch
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append('../')

from src.main.inference import Inference


use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else  "cpu"

main_parser = argparse.ArgumentParser("Generate lip sync video for evaluation")

main_parser.add_argument('--filelist', type=str, help="File path for filelist", default="./test_filelists/lrs3.txt")
main_parser.add_argument('--data_root', type=str,default='/media/peter/peterwish/dataset/lrs3/test/')
main_parser.add_argument('--model_type', type=str, default='attn_lstm')
main_parser.add_argument('--result_dir', type=str, help="Directory to save result", default="./eval_vdo/Lrs3_attn_lstmgen_l1")
main_parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default='/home/peter/Peter/audio-visual/fyp/checkpoints/generator/benchmark/attn_generator_020_l1_1e_2.pth')
main_parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default='../checkpoints/image2image/image2image.pth',required=False)

main_args = main_parser.parse_args()



def call_inference(model_type, gen_ckpt, img2img_ckpt, video, audio, save_path): 

        parser = argparse.ArgumentParser(description="File for running Inference")

        parser.add_argument('--model_type', help='Type of generator model', default=model_type, type=str)

        parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default=gen_ckpt)

        parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default=img2img_ckpt,required=False)

        parser.add_argument('--input_face', type=str, help="File path for input videos/images contain face",default=video, required=False)

        parser.add_argument('--input_audio', type=str, help="File path for input audio/speech as .wav files", default=audio, required=False)

        parser.add_argument('--fps', type=float, help= "Can only be specified only if using static image default(25 FPS)", default=25,required=False)

        parser.add_argument('--fl_detector_batchsize',  type=int , help='Batch size for landmark detection', default = 32)

        parser.add_argument('--generator_batchsize', type=int, help="Batch size for Generator model", default=5)

        parser.add_argument('--output_name', type=str , help="Name and path of the output file", default=save_path)

        parser.add_argument('--vis_fl', type=bool, help="Visualize Facial Landmark ??", default=False)

        parser.add_argument('--test_img2img', type=bool, help="Testing image2image module with no lip generation" , default=False)
        
        parser.add_argument('--only_fl', type=bool, help="Visualize only Facial Landmark ??", default=False)

        args = parser.parse_args()

        eval_result = Inference(args=args)

        eval_result.start()



def main():
  
    data_root = main_args.data_root
    result_folder =  main_args.result_dir
    model_type = main_args.model_type
    gen_ckpt = main_args.generator_checkpoint
    img2img_ckpt = main_args.image2image_checkpoint



    if not os.path.exists(result_folder):
        
        os.mkdir(result_folder)
        
  

    with open(main_args.filelist, 'r') as filelist:

        lines = filelist.readlines()

    for idx, line in enumerate(tqdm(lines)):
        
        

        audio , video = line.strip().split()

        audio = os.path.join(data_root, audio) + '.mp4'

        video = os.path.join(data_root, video) + '.mp4'

        save_path = os.path.join(result_folder, "{}.mp4".format(idx))

        print(save_path)

        print(audio)

        print(video)


        call_inference(model_type,gen_ckpt,img2img_ckpt,video,audio,save_path)

        





main()






