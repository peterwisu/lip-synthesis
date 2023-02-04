
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import face_alignment
from tqdm import tqdm
import numpy as np
from utils.plot import vis_landmark_on_img
from utils import audio
import cv2
import subprocess
import platform
import os
from src.models.image2image import ResUnetGenerator
from src.models.lstmgen import LstmGen as Lip_Gen
from utils.wav2lip import prepare_audio, prepare_video, load_checkpoint
from utils.utils import procrustes
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

device =  "cuda" if use_cuda else "cpu"

class Inference():
    
    
    def __init__ (self, args):
        
        self.fl_batchsize = args.fl_detector_batchsize
        self.gen_batchsize = args.generator_batchsize
        self.image2image_ckpt = args.image2image_checkpoint
        self.generator_ckpt = args.generator_checkpoint
        self.input_face = args.input_face
        self.fps = args.fps
        self.input_audio = args.input_audio
        self.vis_fl = args.vis_fl
        self.output_name = args.output_name
        self.test_img2img = args.test_img2img
        
        
        
        self.all_frames , self.fps = prepare_video(args.input_face, args.fps)
        self.mel_chunk = prepare_audio(args.input_audio, self.fps)
         
        # crop timestamp of a video incase video is longer than audio 
        self.all_frames = self.all_frames[:len(self.mel_chunk)]
        
        
         # Image2Image translation model
        self.image2image = ResUnetGenerator(input_nc=6,output_nc=3,num_downs=6,use_dropout=False)

        # Load pretrained weights to image2image model 
        image2image_weight = torch.load(self.image2image_ckpt)['G']
        # Since the checkpoint of model was trained using DataParallel with multiple GPU
        # It required to wrap a model with DataParallel wrapper class
        self.image2image = DataParallel(self.image2image)
        # assgin weight to model
        self.image2image.load_state_dict(image2image_weight)
        
        self.generator = Lip_Gen().to(device=device)
        
        self.generator = load_checkpoint(model=self.generator,
                        path= self.generator_ckpt,
                        optimizer=None,
                        use_cuda=True,
                        reset_optimizer=True,
                        pretrain=True)

 
    def __landmark_detection__(self,images, batch_size):
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
        

        fls = []
        transforms = []          
        
        for i in tqdm(range(0, len(images), batch_size)):

            img = images[i:i+batch_size]          
            fl_batch = detector.get_landmarks_from_batch(img)

            fl_batch = np.array(fl_batch)[:,:,:] # take only 3d
            
            fl = [] 
            for idx in range(fl_batch.shape[0]):

                fl_inbatch, trans_info = procrustes(fl_batch[idx])
                fl.append(fl_inbatch)
                transforms.append(trans_info)
            
            fl = np.array(fl)

            fls.append(fl)

    
        fls = np.concatenate(fls, axis=0)
        transforms = np.array(transforms)


        return  fls, transforms

    def __keypoints2landmarks__(self,fls):
        """
        
        """
        
        frames = []
        for fl in fls:

            img = np.ones(shape=(256,256,3)) * 255 # blank white image

            fl = fl.astype(int)

            img = vis_landmark_on_img(img,fl).astype(int)
            
            frames.append(img)

        frames = np.stack(frames, axis=0) 

        return frames


    def __reverse_trans__(self,fl , tran):

        scale = tran['scale']
        translate = tran['translate']

        fl = fl *  scale # reverse scaling
        fl = fl + translate # reverse translation
        
        return fl

    def __reverse_trans_batch__ (self, fl , trans) :

        trans_fls =[]
       
        for idx  in range(fl.shape[0]):

            trans_fl = self.__reverse_trans__(fl[idx], trans[idx])

            trans_fls.append(trans_fl)

        trans_fls = np.array(trans_fls)
              
        return trans_fls

    
    def __data_generator__(self):
        """
        
        """

        fl_batch , trans_batch, mel_batch, frame_batch = [],[],[],[]
        
        frames = self.all_frames
        mels  = self.mel_chunk
        
        
        print("Detecting Facial Landmark ....")
        fl_detected, transformation = self.__landmark_detection__(frames, self.fl_batchsize)
        print("Finish detecting Facial Landmark !!!") 

        for i, m in enumerate(mels):

            idx = i % len(frames) # if input if static image only select frame and landmark at index 0
        
            frame_to_trans = frames[idx].copy()
            fl = fl_detected[idx].copy()
            transforms = transformation[idx].copy()
            
            fl_batch.append(fl)
            trans_batch.append(transforms)
            mel_batch.append(m)
            frame_batch.append(frame_to_trans)

            if len(fl_batch) >= self.gen_batchsize:

                fl_batch = np.array(fl_batch)
                trans_batch = np.array(trans_batch) # this might cause error by wrapping a dict in np
                mel_batch = np.array(mel_batch)
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2],1])
                frame_batch = np.array(frame_batch)

                yield fl_batch, trans_batch, mel_batch, frame_batch

                fl_batch, trans_batch, mel_batch, frame_batch = [], [], [], []

        if len(fl_batch) > 0 :

            fl_batch = np.array(fl_batch)
            trans_batch = np.array(trans_batch) # this might cause error by wrapping a dict in np
            mel_batch = np.array(mel_batch)
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2],1])
            frame_batch = np.array(frame_batch)

            yield fl_batch, trans_batch, mel_batch, frame_batch

            fl_batch, trans_batch, mel_batch, frame_batch = [], [], [], []
            
            
    def start(self):
        """
        """
        
        self.data = self.__data_generator__()
        
        if self.vis_fl:
            writer = cv2.VideoWriter('./temp/out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), self.fps, (256*3,256))
        else :
            writer = cv2.VideoWriter('./temp/out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), self.fps, (256,256))
        
        for fl, trans, mel, ref_frame  in self.data:

            # fl shape  (B, 68, 3)
            # mel shape (B, 80, 18, 1)
            # ref frame (B, 256, 256, 3)
            lip_fl =  torch.FloatTensor(fl).to(device)
            lip_fl = lip_fl[:,48:,:] # take only lip keypoints
            lip_fl = lip_fl.reshape(lip_fl.shape[0],-1)
            mel = torch.FloatTensor(mel).to(device)
            mel = mel.reshape(-1,80,18)
            
            if not self.test_img2img: # check if not testing image2image translation module only no lip generator
                with torch.no_grad():

                    self.generator.eval()       
                    out_fl,_ = self.generator(mel, lip_fl, inference=True) 

                out_fl = out_fl.detach().cpu().numpy() # convert output to numpy array
                out_fl = out_fl.reshape(out_fl.shape[0],20,-1)
                
                out_fl = out_fl
                fl[:,48:,:] = out_fl


            fl =  self.__reverse_trans_batch__(fl , trans)

            
            # plot a image of landmarks
            fl_image = self.__keypoints2landmarks__(fl)

        
            # image translation 
            for (img,ref) in zip(fl_image, ref_frame):
                

                
                trans_in = np.concatenate((img,ref), axis=2).astype(np.float32)/255.0
                trans_in = trans_in.transpose((2, 0, 1))
                trans_in = torch.tensor(trans_in, requires_grad=False)
                trans_in = trans_in.reshape(-1, 6, 256, 256) 
                trans_in = trans_in.to(device)
            
                with torch.no_grad():  
                    self.image2image.eval() 
                    trans_out = self.image2image(trans_in)
                    trans_out = torch.tanh(trans_out)
                
                trans_out = trans_out.detach().cpu().numpy().transpose((0,2,3,1)) 
                trans_out[trans_out<0] = 0
                trans_out = trans_out * 255.0
                
                if self.vis_fl: 
                    frame = np.concatenate((ref,img,trans_out[0]),axis=1) 
                else :
                    frame = trans_out[0]
                writer.write(frame.astype(np.uint8))



        # Write video and close writer
        writer.release()
        
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(self.input_audio, 'temp/out.mp4', self.output_name)
        subprocess.call(command, shell=platform.system() != 'Windows')


        
    
        
