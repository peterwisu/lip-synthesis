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

    """

    acc =  accuracy_score(y_pred,y_true, normalize=True) * 100

    return acc


