"""
 
 Inference.py


 This code base of this file was from ***Wav2Lip*** repository but has been modified heavily to perform a lip sythesis via landmarks

 command for testing : python inference.py --input_face ./dummy/00004.mp4 --input_audio ./dummy/audio.wav
"""
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.nn import DataParallel
import face_alignment
from os import listdir
from os import path
import utils.audio as audio
import os
import sys
import time
from src.models import ResUnetGenerator 
from src.models import Lip_Gen
import argparse
import subprocess
import matplotlib.pyplot as plt
from utils.plot import vis_landmark_on_img
import platform
from utils.wav2lip import load_checkpoint



parser = argparse.ArgumentParser(description="File for running Inference")

parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default='checkpoints/generator/checkpoint_lip_fl_epoch000000016.pth')

parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default='./models/ckpt/image2image.pth',required=False)

parser.add_argument('--input_face', type=str, help="File path for input videos/images contain face",default='./dummy/scarlett.jpg', required=False)

parser.add_argument('--input_audio', type=str, help="File path for input audio/speech as .wav files", default='./dummy/audio.wav', required=False)

parser.add_argument('--output_path', type=str, help="Path for saving the result", default='result.mp4', required=False)

parser.add_argument('--fps', type=float, help= "Can only be specified only if using static image default(25 FPS)", default=25,required=False)

parser.add_argument('--fl_detector_batchsize',  type=int , help='Batch size for landmark detection', default = 32)

parser.add_argument('--generator_batchsize', type=int, help="Batch size for Generator model", default=16)

parser.add_argument('--output_name', type=str , help="Name of the output file", default="results.mp4")

parser.add_argument('--vis_fl', type=bool, help="Visualize Facial Landmark ??", default=True)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def landmark_detection(images, batch_size):
    """
    *************************************************************************************** 
    Detect 3D Facial Landmark from images using Landmark Detector Tools from Face_Alignment
    Link repo : https://github.com/1adrianb/face-alignment
    ***************************************************************************************
    @author : Wish Suharitdamrong
    --------
    arguments
    ---------
        images : list of images 
    ------
    return
    ------
    """
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device) 
    images = np.array(images) # shape (Batch , height, width, 3)
    images = np.transpose(images,(0,3,1,2)) # shape (Batch, 3, height, width)
    images = torch.from_numpy(images)
    """
    fls = detector.get_landmarks_from_batch(images)
    fls = np.array(fls)
    """
    
    while 1: 

        fls = []
        
        for i in tqdm(range(0, len(images), batch_size)):

            img = images[i:i+batch_size]          
            fl = detector.get_landmarks_from_batch(img)
            fls.append(fl)

        break

    fls = np.concatenate(fls, axis=0)

    return  fls


def landmark_translation(frames,fl):


    return None



def keypoints2landmarks(fls):
    
    frames = []
    for fl in fls:

        img = np.ones(shape=(256,256,3)) * 255 # blank white image

        fl = fl.astype(int)

        img = vis_landmark_on_img(img,fl).astype(int)
        
        frames.append(img)

    frames = np.stack(frames, axis=0) 

    return frames

def data_generator(frames, mels,fl_batchsize, gen_batchsize):

    fl_batch , mel_batch, frame_batch = [],[],[]
    
    print("Detecting Facial Landmark ....")
    fl_detected = landmark_detection(frames, fl_batchsize)
    print("Finish detecting Facial Landmark !!!") 

    for i, m in enumerate(mels):

        idx = i % len(frames) # if input if static image only select frame and landmark at index 0
    
        frame_to_trans = frames[idx].copy()
        fl = fl_detected[idx].copy()
         
        fl_batch.append(fl)
        mel_batch.append(m)
        frame_batch.append(frame_to_trans)

        if len(fl_batch) >=gen_batchsize:

            fl_batch = np.array(fl_batch)
            mel_batch = np.array(mel_batch)
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2],1])
            frame_batch = np.array(frame_batch)

            yield fl_batch, mel_batch, frame_batch

            fl_batch, mel_batch, frame_batch = [], [], []

    if len(fl_batch) > 0 :

        fl_batch = np.array(fl_batch)
        mel_batch = np.array(mel_batch)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2],1])
        frame_batch = np.array(frame_batch)

        yield fl_batch, mel_batch, frame_batch

        fl_batch, mel_batch, frame_batch = [], [], []



def load_image2image_ckpt(model,path):
    """
    *******************************************
    Load pretrained weight to image2image model
    *******************************************
    @author : Wish Suharitdamrong
    ---------
    arguments
    ---------
        model : image2image translation model 
        path  : path of a model checkpoint 
    ------
    return
    ------
        model :  model with pretrained weights
    """
    
    # load model checkpoint
    ckpt = torch.load(path)['G'] 
    # Since the checkpoint of model was trained using DataParallel with multiple GPU
    # It required to wrap a model with DataParallel wrapper class
    model = DataParallel(model)
    # assgin weight to model
    model.load_state_dict(ckpt)

    return  model

def prepare_video(path):
    """
    *******************************************************************************
    Prepare input image/videos  : Detect a FPS of input and spilt video into frames
    *******************************************************************************
    
    The code in this function was orginially was from ***Wav2Lip*** 

    """
    
    # Checko that the give file exists
    if not os.path.exists(path):

        raise ValueError("Cannot locate a input file in a given path {} in argument --input_face".format(path))
    
    # Check that input face is valid files
    elif not os.path.isfile(path):

        raise ValueError("Input must be valid file at the given path {}  in argument --input_face".format(path))
    
    # Check in input is image
    elif path.split('.')[1] in ['jpg','jpeg','png']:
        # read images  
        all_frames = [cv2.imread(path)]
        # set FPS for inference equal to an input fps
        fps = args.fps

    else:
        # read video
        video = cv2.VideoCapture(path)
        # set FPS for inference equal to FPS of input video
        fps = video.get(cv2.CAP_PROP_FPS)
        all_frames = [] 
        while 1:
            # read each frame of video 
            still_reading, frame= video.read() 
            # if no futher frame stop reading
            if not still_reading:

                video.release() # close video reader
                break
            
            frame = cv2.resize(frame,(256,256)) # resize input image to 256x256 (width and height) 
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            """
            cv2.imshow('asdasd', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            all_frames.append(frame)

    return all_frames, fps

def prepare_audio(path, fps):
    """
    ***************************************************************
    Prepare input audio  : Transform audio input to Melspectrogram 
    ***************************************************************
    
    The code in this function was orginially was from ***Wav2Lip*** 

    """    
    # if the input audio is not .wav file then convert it to .wav
    if not path.endswith('.wav'): 
        # command using ffmpeg to convert audio to .wav and store temporary .wav file {temp/temp.wav}
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(path, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        # change path of the input file to temporary wav file
        path = 'temp/temp.wav'

    wav = audio.load_wav(path, 16000) # load wave audio with sample rate of 16000
    mel = audio.melspectrogram(wav) # transform wav to melspectorgram

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    mel_step_size = 18 #time step in spectrogram
    i = 0
    while 1:

        start_idx = int(i * mel_idx_multiplier)

        if start_idx + mel_step_size > len(mel[0]):

            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break

        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    return mel_chunks

    


def main():

    fl_batchsize = args.fl_detector_batchsize
    gen_batchsize = args.generator_batchsize
    image2image_ckpt = args.image2image_checkpoint

    all_frames, fps = prepare_video(args.input_face)
    mel_chunks = prepare_audio(args.input_audio, fps)
 
    # crop timestamp of a video incase video is longer than audio
    all_frames = all_frames[:len(mel_chunks)]
    
     # put input data into dataloader
    data = data_generator(all_frames, mel_chunks, fl_batchsize, gen_batchsize)
     
    # Image2Image translation model
    image2image = ResUnetGenerator(input_nc=6,output_nc=3,num_downs=6,use_dropout=False)

    # Load pretrained weights to model 
    image2image = load_image2image_ckpt(model=image2image, path= args.image2image_checkpoint)
    
    # Generator
    generator = Lip_Gen().to(device)
    
    generator  = load_checkpoint(model=generator,
                    path=args.generator_checkpoint,
                    optimizer=None,
                    use_cuda=True,
                    reset_optimizer=True,
                    pretrain=True)
    
    # Create video writer
    print(args.vis_fl)
    if args.vis_fl:
        writer = cv2.VideoWriter('./temp/out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), fps, (256*3,256))
    else :
        writer = cv2.VideoWriter('./temp/out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), fps, (256,256))
    
    for fl, mel, ref_frame  in data:

        # fl shape  (B, 68, 3)
        # mel shape (B, 80, 18, 1)
        # ref frame (B, 256, 256, 3)
        lip_fl =  torch.FloatTensor(fl).to(device)
        lip_fl = lip_fl[:,48:,:] # take only lip keypoints
        lip_fl = lip_fl.reshape(-1,60)
        mel = torch.FloatTensor(mel).to(device)
        mel = mel.reshape(-1,80,18)
 
        with torch.no_grad():

            generator.eval()       
            out_fl,_ = generator(mel, lip_fl) 

        out_fl = out_fl.detach().cpu().numpy() # convert output to numpy array
        out_fl = out_fl.reshape(-1,20,3)
        """ 
        print(lip_fl[0].reshape(-1,20,3))
        print(out_fl[0])
        out_fl = out_fl *10
        print(out_fl[0])
        
        exit()
        """
        
        out_fl = out_fl
        fl[:,48:,:] = out_fl
        
        # plot a image of landmarks
        fl_image = keypoints2landmarks(fl)

       
        # image translation 
        for (img,ref) in zip(fl_image, ref_frame):
            
            trans_in = np.concatenate((img,ref), axis=2).astype(np.float32)/255.0
            trans_in = trans_in.transpose((2, 0, 1))
            trans_in = torch.tensor(trans_in, requires_grad=False)
            trans_in = trans_in.reshape(-1, 6, 256, 256) 
            trans_in = trans_in.to(device)
        
            with torch.no_grad():  
                image2image.eval() 
                trans_out = image2image(trans_in)
                trans_out = torch.tanh(trans_out)
            
            trans_out = trans_out.detach().cpu().numpy().transpose((0,2,3,1)) 
            trans_out[trans_out<0] = 0
            trans_out = trans_out * 255.0
            
            if args.vis_fl: 
                frame = np.concatenate((ref,img,trans_out[0]),axis=1) 
            else :
                frame = trans_out[0]
            writer.write(frame.astype(np.uint8))



    # Write video and close writer
    writer.release()
    
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.input_audio, 'temp/out.mp4', args.output_name)
    subprocess.call(command, shell=platform.system() != 'Windows')



    # image translation model
    
if __name__ == '__main__':

    main()

    
