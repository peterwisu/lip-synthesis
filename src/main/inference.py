
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
#from src.models.lstmgen import LstmGen as Lip_Gen
#from src.models.lstmattn import LstmGen as Lip_Gen

#from src.models.transgen import TransformerGenerator  as Lip_Gen
from utils.wav2lip import prepare_audio, prepare_video, load_checkpoint
from utils.utils import procrustes
import matplotlib.pyplot as plt
from utils.plot import plot_scatter_facial_landmark

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
        self.seq_len = 5#args.seq_len
        self.model_type = args.model_type
        
        
        
        
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



        if self.model_type == "lstm":

            from src.models.lstmgen import LstmGen as Lip_Gen

            print("Import LSTM generator")
            
        elif self.model_type == "attn_lstm":

            from src.models.lstmattn import LstmGen as  Lip_Gen

            print("Import Attention LSTM generator")

 
        
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
        
        def detect_bug_136(fls):
            """
            Some times when using detector.get_landmarks_from_batch it does has some bug. Instead of returning facial landmarks (68,3) for single person in image it instead
            return (136,3) or (204,3). The first 68 point still a valid facial landamrk of that image (as I visualised). So this fuction basically removed the extra 68 point in landmarks.
            """
            
            
            for i in range(len(fls)):
                
                print(np.array(fls[i]).shape)
                if len(fls[i]) != 68:
                    
                    bug = fls[i]
            
                    fl1 = bug[:68]
                    
                    #fl2 = bug[68:]
                    
                 
                    
                    fls[i] = fl1
                if len(fls[i]) == 0:
                    
                    fls[i] = fls[i-1]
                
        
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

          
            
            detect_bug_136(fl_batch)
            
                
            
            fl_batch = np.array(fl_batch)#[:,:,:] # take only 3d
            
           
            
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

        fl_seq , trans_seq, mel_seq, frame_seq = [],[],[],[]
        
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
            
            fl_seq.append(fl)
            trans_seq.append(transforms)
            mel_seq.append(m)
            frame_seq.append(frame_to_trans)
            

            if len(fl_seq) >= self.seq_len: 

                fl_batch.append(fl_seq)
                trans_batch.append(trans_seq)
                mel_batch.append(mel_seq)
                frame_batch.append(frame_seq)

                fl_seq , trans_seq, mel_seq, frame_seq = [],[],[],[]

        
            if len(fl_batch) >= self.gen_batchsize:

                fl_batch = np.array(fl_batch)
                trans_batch = np.array(trans_batch) # this might cause error by wrapping a dict in np
                mel_batch = np.array(mel_batch)
                mel_batch = np.reshape(mel_batch, [len(mel_batch), self.seq_len , 1 , mel_batch.shape[2], mel_batch.shape[3]]) # b ,s ,1 , 80 , 18  (old 80,18,1)
                frame_batch = np.array(frame_batch)
              


                yield fl_batch, trans_batch, mel_batch, frame_batch

                fl_batch, trans_batch, mel_batch, frame_batch = [], [], [], []
        
        #print(np.array(fl_batch).shape)
        #print(np.array(fl_seq).shape)

    
        if len(fl_batch) > 0 :
            #print("tt")
            fl_batch = np.array(fl_batch)
            #print(fl_batch.shape)
            trans_batch = np.array(trans_batch) # this might cause error by wrapping a dict in np
            #print(trans_batch.shape)
            mel_batch = np.array(mel_batch)
            #print(mel_batch.shape)
            mel_batch = np.reshape(mel_batch, [len(mel_batch), self.seq_len,1 ,mel_batch.shape[2], mel_batch.shape[3]])
            #print(mel_batch.shape)
            frame_batch = np.array(frame_batch)
            
            yield fl_batch, trans_batch, mel_batch, frame_batch

            fl_batch, trans_batch, mel_batch, frame_batch = [], [], [], []

        if len(fl_seq) > 0:

            #print("hello")


            fl_batch = np.expand_dims(np.array(fl_seq),axis=0)
            #print(fl_batch.shape)
            trans_batch = np.expand_dims(np.array(trans_seq),axis=0) # this might cause error by wrapping a dict in np
            #print(trans_batch.shape)
            mel_batch = np.expand_dims(np.array(mel_seq),axis=0)
            curr_mel_seq = mel_batch.shape[1]
            #print(mel_batch.shape)
            mel_batch = np.reshape(mel_batch, [len(mel_batch), curr_mel_seq,1 ,mel_batch.shape[2], mel_batch.shape[3]])
            #print(mel_batch.shape)
            frame_batch = np.expand_dims(np.array(frame_seq),axis=0)
            
            #exit()

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
        
        for (fl, trans, mel, ref_frame) in tqdm(self.data):
            
            # fl shape  (B, 68, 3)
            # mel shape (B, 80, 18, 1)
            # ref frame (B, 256, 256, 3)
            lip_fl =  torch.FloatTensor(fl).to(device)

            lip_fl = lip_fl[:,:,48:,:] # take only lip keypoints
    
            lip_seq = lip_fl.size(0)
            lip_fl = torch.stack([lip_fl[0] for _ in range(lip_seq)], dim=0)
            lip_fl = lip_fl.reshape(lip_fl.shape[0],lip_fl.shape[1],-1)
            mel = torch.FloatTensor(mel).to(device)
            #print(mel.shape)
            #mel = mel.reshape(-1,80,18)
            
            if not self.test_img2img: # check if not testing image2image translation module only no lip generator
                with torch.no_grad():

                    self.generator.eval()       
                    out_fl,_ = self.generator(mel, lip_fl, inference=False) 
                
            
                out_fl = out_fl.detach().cpu().numpy() # convert output to numpy array
                out_fl = out_fl.reshape(out_fl.shape[0],out_fl.shape[1],20,-1)
                
                out_fl = out_fl
                fl[:,:,48:,:] = out_fl
            

            fl = fl.reshape(-1,fl.shape[2],fl.shape[3])
            #ref_frame = ref_frame.reshape(-1,ref_frame.shape[2], ref_frame[3])
            trans = trans.reshape(-1)
            fl =  self.__reverse_trans_batch__(fl , trans)

            
            # plot a image of landmarks
            fl_image = self.__keypoints2landmarks__(fl)

            
            fl_image = fl_image.reshape(ref_frame.shape[0],ref_frame.shape[1],ref_frame.shape[2],ref_frame.shape[3],ref_frame.shape[4])

                   
            # image translation 
            for (img_batch,ref_batch) in zip(fl_image, ref_frame):
                
                for img, ref in zip(img_batch, ref_batch): 

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


        
    
        
