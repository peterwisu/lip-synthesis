"""


This file contain a Utility function from ***Wav2lip***


Links : https://github.com/Rudrabha/Wav2Lip


"""

from os.path import join
import torch
import os
from torch.nn import DataParallel
import cv2
import subprocess
import numpy as np
from utils import audio


def save_checkpoint(model, optimizer,  checkpoint_dir,epoch, savename):
    checkpoint_path = join(
        checkpoint_dir, savename)

    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoints:", checkpoint_path)


def _load(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer,use_cuda, reset_optimizer=False, pretrain=False):

    global global_epoch

    print("Load checkpoints from: {}".format(path))

    checkpoint = _load(path, use_cuda)
    try :
        model.load_state_dict(checkpoint["state_dict"])
    except Exception as e:
        
        model = DataParallel(model)
        model.load_state_dict(checkpoint["state_dict"])
        
    if not pretrain:
        if not reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                print("Load optimizer state from {}".format(path))
                optimizer.load_state_dict(checkpoint["optimizer"])

    if not pretrain:
        global_epoch = checkpoint["global_epoch"]
        return model, optimizer, global_epoch
    
    else:
        
        return model
    


# This function originally name **get_image_list**
def get_fl_list(data_root, split):

	filelist = []
	with open('filelists/{}.txt'.format(split)) as f:
		for line in f:
			line = line.strip()
			if ' ' in line: line = line.split()[0]
			filelist.append(os.path.join(data_root, line))

	return filelist





################




def prepare_video(path,in_fps):
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

        img  = cv2.imread(path)
        img  = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
        print(img.shape)

        # read images  
        all_frames = [img]
        # set FPS for inference equal to an input fps
        fps = in_fps

    else:
        print( path.split('.') )
        print("vido")
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

    




