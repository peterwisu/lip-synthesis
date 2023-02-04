"""

This code is originally from  ***Wav2Lip*** repository

link : https://github.com/Rudrabha/Wav2Lip

This code has been modified to load a datasets containing Facial Landmarks instead face images

"""

from hparams import hparams
from utils.wav2lip import get_fl_list
from os.path import basename, isfile, dirname, join
#import cv2
import os
import utils.audio as audio
import time 
import random
from glob import glob
import torch
import numpy as np
import warnings
from utils.utils import procrustes
from utils.plot import plot_scatter_facial_landmark , plot_lip_landmark



warnings.simplefilter(action='ignore', category=FutureWarning)
syncnet_T = 5
syncnet_mel_step_size = 16


class Dataset(object):
    def __init__(self, split, args):

        self.split = split
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

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            
            fl = np.loadtxt(fname) #  load facial landmark at that particular frame

            # check whether is facial landmark can be frontalize or not?
            try:


                #fl = fl[48:,:] # take only lip
        
                #norm_lip, _ =   procrustes(lip) # normalize only lip

                #fl[48:,:] = norm_lip # note that in the fl array containing 3d facial landamrks but lip are in normalize form 
                fl ,_ =procrustes(fl)
                fl = fl[48:,:]
                
            except Exception as e:
                
                fl = None
            
            if fl is None:
                return None

            window.append(fl)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size


        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0:
            return None

        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)
            

        mels = np.asarray(mels)

        return mels

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):

        
        while 1:

            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            self.vidname=vidname

        
            img_names = list(glob(join(vidname, '*.txt')))
        
            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if mel.shape[0] != syncnet_mel_step_size:

                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None:

                continue

            y = np.array(window.copy())

            x = np.array(wrong_window)
            x = torch.FloatTensor(x[:, :, :]) 
            x =  [x[0] for i in range(x.size(0))] 
            x =  torch.stack(x , dim=0)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y[:,:, :])

            return x, indiv_mels, mel, y

   

