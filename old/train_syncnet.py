"""

 THis file contains few of a code from wav2lip


"""
from tkinter import image_names
import math
from traceback import print_tb
import warnings
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models import SyncNet_fl as SyncNet
from models import modSync
import utils.audio as audio
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os, random, cv2, argparse
from hparams import hparams
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from dataset import SyncNet_Dataset
from utils.wav2lip import save_checkpoint, load_checkpoint
from utils.utils import save_logs, load_logs
from utils.plot import plot_comp, plot_roc, plot_cm
from utils.func_utils import get_accuracy

warnings.simplefilter(action='ignore', category=FutureWarning)
parser = argparse.ArgumentParser(description="Code for training SyncNet with the lip key points ")

parser.add_argument("--data_root", help="Root for preprocessed lip keypoint dataset", default='/home/peter/Peter/audio-visual/lrs2_pretrainfl_256_full_face_prepro')

parser.add_argument("--checkpoint_dir", help="dir to save checkpoints for SyncNet for lip keypoint", default='./checkpoints/syncnet/',
                    type=str)

parser.add_argument('--checkpoint_path', help="Resume from checkpoints ", default=None, type=str)

parser.add_argument('--save_name', help="name of a save", default="syncnet_test_pretrain",type=str)

args = parser.parse_args()


global_epoch = 0
use_cuda = torch.cuda.is_available()

print('use_cuda : {}'.format(use_cuda))

BCE = nn.BCELoss()
sigmoid = nn.Sigmoid()

def cosine_loss(audio,lip, y):
    
    distance = nn.functional.cosine_similarity(audio, lip)
 
    # Get a probabilites predtict of each dataset
    proba = distance.detach().clone().cpu().numpy() # copy a array of a probalities
    

    pred = [1 if y >= 0.5 else 0 for y in distance]
    pred = np.array(pred)

    correct = 0

    for predict, actual in zip(pred, y):
        if predict == actual:
            correct += 1

    accuracy = get_accuracy(pred,y.detach().clone().cpu().numpy())#100 * (correct / len(y))

  
    loss = BCE(distance.unsqueeze(1), y)
    


    return loss, accuracy, pred, proba


def train(device,
          model,
          train_data_loader,
          test_data_loader,
          optimizer,
          save_name,
          checkpoint_dir=None,
          nepochs=None,
          continue_ckpt=False,
          ):

    global  global_epoch


    if continue_ckpt:

        print('Starting Checkpoint from epochs :', global_epoch)
        
        train_loss, train_acc, vali_loss, vali_acc = load_logs(model_name="syncnet",savename="{}.csv".format(save_name), epoch=global_epoch,type_model='syncnet')

        global_epoch += 1
        

    else:

        print("Not Continue from Checkpoint")
        train_loss = np.array([])
        vali_loss = np.array([])
        train_acc = np.array([])
        vali_acc = np.array([])

    while global_epoch < nepochs:

        running_loss = 0.
        running_acc = 0.
        iter_inbatch = 0
        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (x, mel, y) in prog_bar:

            # X shape : (B, 5 , 20 ,3)
            # MEl shape : (B,1 , 80, 16 )

            model.train()
            optimizer.zero_grad()
            # allocate data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            
            
            a, v = model(mel, x)
            y = y.to(device)
             
            loss, accuracy_in_batch, _, _ = cosine_loss(a, v, y)

            
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()
            running_acc += accuracy_in_batch
            iter_inbatch += 1

            prog_bar.set_description('TRAIN EPOCHS : {} Loss: {} Accuracy: {}'.format(global_epoch,
                                                                                running_loss / iter_inbatch,
                                                                                running_acc / iter_inbatch))


        print("**************************************************")

        

        vali_cur_loss, vali_cur_acc, cm_fig, roc_fig= eval_model(test_data_loader, device, model, global_epoch)


        # append logs
        vali_loss = np.append(vali_loss, vali_cur_loss)
        vali_acc = np.append(vali_acc, vali_cur_acc)
        train_loss = np.append(train_loss, running_loss / iter_inbatch)
        train_acc = np.append(train_acc, running_acc/iter_inbatch)
        
        # save logs
        save_logs(train_loss=train_loss, train_acc=train_acc, vali_loss=vali_loss, vali_acc=vali_acc,
                  model_name="syncnet",savename='{}.csv'.format(save_name))

        # plot metrics
        loss_comp  = plot_comp(train_loss, vali_loss,  name="Loss")
        acc_comp = plot_comp(train_acc, vali_acc, name="Accuracy")


        # ********Tensorboard***********
        
        # Scalar
        writer.add_scalar("Optim/Lr", optimizer.param_groups[0]['lr'],global_epoch)
        writer.add_scalar('Loss/train', running_loss/iter_inbatch, global_epoch)
        writer.add_scalar('Loss/vali', vali_cur_loss, global_epoch)
        writer.add_scalar('Acc/train', running_acc/iter_inbatch, global_epoch)
        writer.add_scalar('Acc/vali', vali_cur_acc, global_epoch)

        # Figure
        writer.add_figure('Plot/loss', loss_comp, global_epoch)
        writer.add_figure('Plot/acc', acc_comp, global_epoch)
        writer.add_figure("Vis/confusion_matrix", cm_fig, global_epoch)
        writer.add_figure("Vis/ROC_curve", roc_fig, global_epoch)


        # save checkpoint
        save_checkpoint(model, optimizer, checkpoint_dir, global_epoch)
        global_epoch += 1


def eval_model(test_data_loader, device, model,epoch):
    
    losses = []
    accuracy = []
    
    # array for visualizing CM and ROC
    y_pred_label = np.array([])
    y_pred_proba = np.array([])
    y_gt =  np.array([])
    
    with torch.no_grad():
        for step, (x, mel, y) in tqdm(enumerate(test_data_loader)):

            model.eval()
         
            # Transform data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)
            
            a, v  = model(mel, x)
         
            loss, accuracy_in_eval, pred_label, pred_proba = cosine_loss(a, v, y)

            # add label and proba to a array
            y_pred_label = np.append(y_pred_label, pred_label)
            y_pred_proba = np.append(y_pred_proba, pred_proba)
            y_gt   = np.append(y_gt, y.clone().detach().cpu().numpy())

            losses.append(loss.item())
            accuracy.append(accuracy_in_eval)

        # plot roc and cm
        roc_fig = plot_roc(y_pred_proba,y_gt)
        cm_fig =  plot_cm(y_pred_label,y_gt)

        averaged_loss = sum(losses) / len(losses)
        averaged_accuracy = sum(accuracy) / len(accuracy)

    """
    debug
    """
    """ 
    iterdata = iter(test_data_loader)
    x, mel, y = iterdata.next()
    with torch.no_grad():
        model.eval()
        x = x.to(device)
        mel = mel.to(device)
        a, v = model(mel, x)
        
        distance = nn.functional.cosine_similarity(a,v)
    """
    """        
    print(a[0])
    print(v[0])
    print(distance[0])
    print(y[0])
    """

    print("VALI EPOCHS : {} : LOSS: {} , ACC: {}".format(epoch, averaged_loss , averaged_accuracy))
    print("************************************************")

    return averaged_loss, averaged_accuracy ,cm_fig, roc_fig




if __name__ == "__main__":

    save_name = args.save_name
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    batch_size = hparams.syncnet_batch_size * torch.cuda.device_count()

    writer = SummaryWriter("../tensorboard/{}".format(save_name))


    print("Running save name :  {}".format(save_name))

    # if create checkpoint dir if it does not exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = SyncNet_Dataset(split='pretrain', args=args)
    test_dataset = SyncNet_Dataset(split='test', args=args)

    train_data_loader = data_utils.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=hparams.num_workers)

    print("Training dataset {}".format(len(train_dataset)))
    print("Testing dataset {}".format(len(test_dataset)))


    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    #model = SyncNet().to(device)
    model = modSync().to(device)


    #dataiter = iter(test_data_loader)
    #x, mel ,_= dataiter.next()
    

    # print(x.shape)
    # print(mel.shape)
    # writer.add_graph(model, (mel.to(device), x.to(device)))
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr,
                           )


    continue_ckpt = False
    if checkpoint_path is not None:
        continue_ckpt = True
        model, optimizer, global_epoch = load_checkpoint(checkpoint_path, model, optimizer,use_cuda, reset_optimizer=False)
 
    # Parallel Training on multiple GPU
    if torch.cuda.device_count() > 1:

        model = DataParallel(model)

    model.to(device)

    

    print("Start Training")
    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          nepochs=hparams.nepochs,
          continue_ckpt=continue_ckpt,
          save_name=save_name)

    writer.close()
