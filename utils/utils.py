"""

****This file contain utlities functions****

@author : Wish Suharitdamrong

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from sklearn.metrics import accuracy_score


def save_logs(model_name,savename,**kwargs):
    """
    *********
    save_logs : Save a logs in .csv files
    *********
    @author: Wish Suharitdamrong
    ------
    inputs : 
    ------ 
        model_name : name of a model
        savename :  name of a file saving as a .csv
        **kwagrs : array containing values of metrics such as accuracy and loss 
    -------
    outputs :
    -------
    """
    
    path = "./logs/{}/".format(model_name)

    if not os.path.exists(path):
        os.mkdir(path)
    
    df = pd.DataFrame()
    for log in kwargs:

        df[log] = kwargs[log]

    savepath = os.path.join(path,savename)

    df.to_csv(savepath, index=False)


def load_logs(model_name,savename, epoch, type_model=None):
    """
    *********
    load_logs : Load a logs file from csv 
    *********
    @author: Wish Suharitdamrong
    ------
    inputs : 
    ------
        model_name : name of a model
        savename : name of a logs in .csv files
        epoch :  number of iteration continue from checkpoint
    -------
    outputs :
    -------
        train_loss : array containing training loss
        train_acc : array containing training accuracy
        vali_loss : array containing validation loss
        vali_acc : array containing validation accuracy
    """

    path = "./logs/{}/".format(model_name)
   
    savepath = os.path.join(path,savename)
    
    if type_model is None:
        
        raise ValueError("Type of model should be specified Generator or SyncNet")

    if not os.path.exists(savepath):

        print("Logs file does not exists !!!!")

        exit()
     
    df =  pd.read_csv(savepath)[:epoch+1]
    
    if type_model == "syncnet":

        train_loss = df["train_loss"]
        train_acc = df['train_acc']
        vali_loss = df['vali_loss']
        vali_acc = df['vali_acc']

        
        return train_loss, train_acc, vali_loss, vali_acc
    
    elif type_model == "generator":
        
        train_loss = df["train_loss"]
        vali_loss = df["vali_loss"]
        
        return train_loss, vali_loss
    
    else :
        
        
        raise ValueError(" Argument type of model (type_model) should be either 'generator' or 'syncnet' !!!!")
    
    
    


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def norm_lip2d(fl_lip, distance=None):
  
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


def get_accuracy(y_pred,y_true):
    """
    *********
    get_accuracy : calcualte accuracy of a model
    *********
    @author: Wish Suharitdamrong
    ------
    inputs : 
    ------
        y_pred : predicted label
        y_true : ground truth of a label
    -------
    outputs :
    -------
        acc : accuracy of a model
 
    """

    acc =  accuracy_score(y_pred,y_true, normalize=True) * 100

    return acc


def procrustes(fl):
    
    transformation = {}

    fl, mean = translation(fl)

    fl, scale = scaling(fl)

    #fl , rotate = rotation(fl)

    transformation['translate'] = mean 

    transformation['scale'] = scale

    #transformation['rotate'] = rotate

    return  fl , transformation


def translation(fl):

    mean =  np.mean(fl, axis=0)

    fl = fl - mean

    return fl , mean

def scaling(fl):

    scale = np.sqrt(np.mean(np.sum(fl**2, axis=1)))

    fl = fl/scale

    return  fl , scale

def rotation(fl): 

    left_eye , right_eye = get_eye(fl)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    dz = right_eye[2] - left_eye[2]
    
    #  Roll : rotate through z axis 
    if dx!=0 :
        f = dy/dx
        a = np.arctan(f)
        roll = np.array([
                    [math.cos(a), -math.sin(a)  , 0],
                    [math.sin(a), math.cos(a)   , 0],
                    [0,               0         , 1]
                ])
    else:
        roll = np.array([
                    [1 , 0 , 0],
                    [0 , 1 , 0],
                    [0 , 0 , 1]
                ])

    # Yaw : rotate through y axis        
    f = dx/dz

    a = np.arctan(f)
    yaw = np.array([
                [math.cos(a),0, math.sin(a)],
                [0,1,0],
                [-math.sin(a),0,math.cos(a)]
    ])

   # 
   # f=  dz/dy
   # a = np.arctan(f)
   # pitch = np.array([
   #             [1,0,0],
   #             [0,math.cos(a), -math.sin(a)],
   #             [0,math.sin(a),math.cos(a)],
   #            
   # ])
    
    # Roate face in  frontal pose
    a = np.arctan(90)
    frontal = np.array([
                [math.cos(a),0, math.sin(a)],
                [0,1,0],
                [-math.sin(a),0,math.cos(a)]
    ])
    
    #  transformation for rotation
    rotate = np.matmul(np.matmul(roll,yaw),frontal)

    fl = np.matmul(fl,rotate)
 

    return fl , rotate

def get_eye(fl):
    """
    get_eye : get center of both eye on the facial landmarks
    """

    left_eye  = np.mean(fl[36:42,:], axis=0)
    right_eye = np.mean(fl[42:48,:], axis=0)

    return left_eye, right_eye



    


