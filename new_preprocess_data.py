"""
 This code originally from ***Wav2Lip*** repository

 Link repo: https://github.com/Rudrabha/Wav2Lip

 This code has been modified to preprocess Facial Landmark dataset instead of face images

"""

import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import utils.audio as audio
from hparams import hparams as hp
import face_alignment

import torch
import matplotlib.pyplot as plt
from utils.plot import vis_landmark_on_img

print('Running Preprocess FL ')
parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=16, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset" , default="/media/peter/peterwish/dataset/lrs2_v1/mvlrs_v1/main/")
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default="../lrs2_test/")

args = parser.parse_args()

fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, 
                                    device='cuda:{}'.format(id)) for id in range(args.ngpu)]



fa_landmark = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'




def detect_bug_fl_batch(fls):

    for i in range(len(fls)):

        if len(fls[i]) > 68:

            bug = fls[i]

            fl = bug[:68]

            fls[i] = fl

    return fls

    


def process_video_file(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)
    
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        #print(np.array(frames).shape)
        frame = cv2.resize(frame, (256,256))
        frames.append(frame)
    
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    i = -1
    
    for batch in batches:
        
        batch = np.array(batch)

        batch = np.transpose(batch, (0,3,1,2))

        batch = torch.from_numpy(batch)
        
        preds = fa[gpu_id].get_landmarks_from_batch(batch)

        preds = detect_bug_fl_batch(preds)

        for j, fl in enumerate(preds):

            i +=1

            if len(fl) == 0:

                continue

            np.savetxt(path.join(fulldir,'{}.txt'.format(i)),fl ,fmt='%.4f')

            

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)

    
def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))
    
    print("Saving Path at {}".format(args.preprocessed_root))

    filelist = glob(path.join(args.data_root, '*/*.mp4'))

    jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]

    
    p = ThreadPoolExecutor(args.ngpu)


    futures = [p.submit(mp_handler, j) for j in jobs]

    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print('Dumping audios...')

    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main(args)
