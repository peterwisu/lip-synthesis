"""

This code is originally from ***Wav2Lip*** repository 

Link : https://github.com/Rudrabha/Wav2Lip

The code have been modified to load a dataset containing Facial Landmarks instead of face images

"""
import os
from os.path import dirname, join, basename, isfile
import random
from hparams import hparams
from glob import  glob
import argparse
import torch
import numpy as np
import utils.audio as audio
import math
import matplotlib.pyplot as plt
from utils.wav2lip import get_fl_list
from utils.front import frontalize_landmarks
import warnings




syncnet_T = 5
syncnet_mel_step_size = 18

front_weight = np.load('./checkpoints/front/frontalization_weights.npy')



class Dataset(torch.utils.data.Dataset):

    def __init__(self, split, args):

        self.all_videos = get_fl_list(args.data_root, split)

    def get_frame_id(self, frame):

        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):

        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []

        for frame_id in range(start_id, start_id + syncnet_T):

            frame = join(vidname, '{}.txt'.format(frame_id))

            if not isfile(frame):
                return None
            window_fnames.append(frame)

        return window_fnames

    # Get audio of five frame
    def crop_audio_window(self, spec, start_frame):

        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx:end_idx, :]


    def visualize_lip(fl ,Point=False, cv=True):
        temp = fl

        fl =temp 
        
        fl[:,1] = -fl[:,1] 
        fig = plt.figure()
        
        ax = fig.add_subplot()
        ax.scatter(fl[:,0],fl[:,1],s=20, c='r',linewidths=4)
        ax.plot(fl[0:7,0],fl[0:7,1], c="tab:pink", linewidth=3 )
        ax.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:pink", linewidth=3 )
        ax.plot(fl[12:17,0],fl[12:17,1], c="tab:pink", linewidth=3 )
        ax.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:pink", linewidth=3)
        ax.axvline(x = 1, color = 'b', linestyle='--',label = 'axvline - full height')
        fig.tight_layout() 
        plt.show()
        return None

    def visualize_face(self, fl):

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(fl[:,0],fl[:,1],s=20,c='r')
        fig.tight_layout()
        plt.show()

            
    def norm_lip2d(self, fl_lip, distance=None):
      
        if distance is None:
        
            distance_x =  fl_lip[6,0] - fl_lip[0,0]
            distance_y =  fl_lip[6,1] - fl_lip[0,1]
     
            distance   =  math.sqrt(pow(distance_x,2)+pow(distance_y,2))
       
            fl_x=(fl_lip[:,0]-fl_lip[0,0])/distance
            fl_y=(fl_lip[:,1]-fl_lip[0,1])/distance


        else:
        
            fl_x=(fl_lip[:,0]-fl_lip[0,0]) / distance
            fl_y=(fl_lip[:,1]-fl_lip[0,1]) / distance
        

        return  np.stack((fl_x,fl_y),axis=1) , distance


    # Lenght of dataset
    def __len__(self):

        return len(self.all_videos)

    def __getitem__(self, idx):

        while 1:

            # Random idx between len of dataset

            idx = random.randint(0, len(self.all_videos) - 1)

            # get a video at idx

            vidname = self.all_videos[idx]


            # get all image from dir vidname (11231/*.txt) in a list

            fl_names = list(glob(join(vidname, '*.txt')))

            if len(fl_names) <= 3 * syncnet_T:
                continue

            fl_name = random.choice(fl_names)

            wrong_fl_name = random.choice(fl_names)

            while wrong_fl_name == fl_name:
                wrong_fl_name = random.choice(fl_names)

            # choose which image to be x or y

            if random.choice([True, False]):

                y = torch.ones(1).float()
                chosen = fl_name

            else:

                y = torch.zeros(1).float()
                chosen = wrong_fl_name

            # get all the fl path of each frame in vdo

            window_fnames = self.get_window(chosen)

            if window_fnames is None:
                continue

            # load  all frame of vdo

            window = []
            temp_dis = 0
            all_read = True
             
            for i, fname in enumerate(window_fnames):

                rfl = np.loadtxt(fname)[:,:2]
                                
                try :
                    fl = frontalize_landmarks(rfl,front_weight)

                except Exception as e :
                    
                    all_read = False
                
                    fl = rfl
                    break
            
                
                
                fl = fl[48:,:] #  take only lip
            
                if fl is None:
                    all_read = False
                    break

                if i == 0:

                    
                    fl, temp_dis = self.norm_lip2d(fl)
                    
                
                else:

                    
                    fl, _ = self.norm_lip2d(fl,temp_dis)


                window.append(fl)


            if not all_read: continue

        

            # get audio

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T #(timestep, 80)
            except Exception as e:
                continue

            

            # get audio of 5 frames

            mel = self.crop_audio_window(orig_mel.copy(), fl_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            window = np.array(window).reshape(5, -1)
            x = torch.FloatTensor(window)

            mel = torch.FloatTensor(mel.T).unsqueeze(0)

        
            return x, mel, y
