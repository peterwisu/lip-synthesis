import matplotlib.pyplot as plt
from glob import glob
from utils.wav2lip import get_fl_list
import random
from os.path import join
import numpy as np
from tqdm import tqdm
from utils.utils import procrustes

def cal_mean(fl):

    return np.mean(fl)



data_root = '../dataset/lrs2_main_fl_256_full_face_prepro/'

video_list = get_fl_list(data_root,'train')


fl_mean = []

trans_mean = []


for _ in tqdm(range(len(video_list))):



    # randomly select video name 
    while True:
        try:
            random_vdo = random.choice(list(glob(join(random.choice(video_list), '*.txt'))))
            break
        except Exception as e :
            print(e)
            continue
    

    fl = np.loadtxt(random_vdo)

    fl_mean.append(cal_mean(fl))

    trans_fl, _ = procrustes(fl)

    trans_mean.append(cal_mean(trans_fl))



#print(fl_mean)

plt.hist(fl_mean, bins=50)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
plt.show()



plt.hist(trans_mean, bins=50)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
plt.show()


    
