import os.path

from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils import data as dataset
import numpy as np
import matplotlib.pyplot as plt
# from models import Lip_Gen
from src.models.generator import Lip_Gen
from hparams import hparams
from src.dataset.generator import Dataset
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
from models import SyncNet_fl as Syncnet
from os.path import join
from utils.func_utils import BatchLipNorm
from utils.plot import plot_compareLip, plot_visLip
from utils.wav2lip import load_checkpoint , save_checkpoint
from utils.utils import save_logs,load_logs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Code for training a lip sync generator via landmark')


""" ---------- Dataset --------"""
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/home/peter/Peter/audio-visual/lrs2_fl_full_face_prepro/', type=str)

""" --------- Generator --------"""
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='./checkpoints/generator/', type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoints', default=None, type=str)

"""---------- SyncNet ----------"""
# path of pretrain syncnet weight
parser.add_argument('--checkpoint_syncnet_path', help="Checkpoint for pretrained SyncNet", default='./models/ckpt/syncnet_fl.pth' ,type=str)

"""---------- Save name --------"""
parser.add_argument('--save_name', help='name of a save', default="generator_mse_without_syncnet", type=str)


args = parser.parse_args()

BCE_LOSS = nn.BCELoss()

blnorm = BatchLipNorm()

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
"""
def _load(checkpoint_path):

    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoints from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model
"""



def get_sync_loss(audio, lip, syncnet, i, eval=False):
    """
    *************
    Get_sync_loss : 
    *************
    @author : Wish Suharitdamrong
    ---------
    arguments : 
    ---------
    -------
    returns
    -------
    """
    
    # reshape a lip in  a shape of (BATCH SIZE, 5(five consecutive landmark), 20 (for 20 keypoints of lip), 3 (for three coordinates x, y, z))
    lip = lip.reshape(-1,5,20,3) 
    # normalize sequnce of a lip
    lip_fl = blnorm(lip)
 
    # visualize lip on tensorboard
    if i == 1 and eval:
        
        for i in range(5):
            du = plot_visLip(lip_fl[0,i].detach().clone().cpu().numpy())
            writer.add_figure('Lip/norm',du,i)
    
    syncnet.eval()

    audio_embedding, lip_embedding = syncnet(audio, lip_fl)

    #print(audio_embedding[0]) 
     
    # calculate distance between two vector embedding
    distance = nn.functional.cosine_similarity(audio_embedding, lip_embedding)
    #print(distance)

    # calculate loss
    y = torch.ones(lip_fl.shape[0], 1).to(device)
    loss = BCE_LOSS(distance.unsqueeze(1), y)


    return loss




def train(device,
          generator,
          syncnet,
          train_dataset,
          validation_dataset,
          optimizer,
          save_name,
          checkpoint_dir=None,
          nepochs=None,
          continue_ckpt=False):

    global global_epoch

    if continue_ckpt:

        print("Starting Checkpoint from epochs : ",global_epoch)

        train_loss , vali_loss = load_logs(model_name="generator", savename="{}.csv".format(save_name),epoch=global_epoch, type_model='generator')

        global_epoch +=1

    else:
        
        global_epoch = 0

        print("Not continue form Checkpoint")
        train_loss = np.array([])
        vali_loss = np.array([]) 

    while global_epoch < nepochs:

        print("Training at Epochs {}".format(global_epoch))
        running_sync_loss = []
        running_l1_loss = []
        runnig_laplacian_loss = []
        running_loss = []
        iter_in_batch = 0
        progress_bar = tqdm(enumerate(train_dataset))

        for _, (con_lip, seq_mels, mel, gt_lip) in progress_bar:

            optimizer.zero_grad()
            generator.train()

            con_lip = con_lip.to(device)
            seq_mels = seq_mels.to(device)
            mel = mel.to(device)
            gt_lip = gt_lip.to(device)

            gen_lip, _ = generator(seq_mels, con_lip)

            gt_lip = gt_lip.reshape(gen_lip.shape[0], -1, 60)

            # L1 loss
            l1_loss = 0
            for idx in range(gen_lip.shape[1]):

                #l1_loss += nn.functional.l1_loss(gen_lip[:, idx, :], gt_lip[:, idx, :])
                l1_loss += nn.functional.mse_loss(gen_lip[:, idx, :], gt_lip[:, idx, :])

            l1_loss = l1_loss/5
            
            if False: #global_epoch > 10 : 
                sync_loss =  get_sync_loss(mel, gen_lip, syncnet, iter_in_batch)

            else:
                sync_loss = torch.tensor(0)
            alpha = 0.8
            beta = 0.2
            if False:# global_epoch > 10:
                loss = l1_loss+ sync_loss
            else:
                loss = l1_loss 
            loss.backward()
            optimizer.step()

            running_l1_loss.append(l1_loss.item())
            # runnig_laplacian_loss.append(laplacian_loss.item())
            running_sync_loss.append(sync_loss.item())
            running_loss.append(loss.item())

            iter_in_batch +=1
            
            progress_bar.set_description('Epochs : {} ,Total loss : {} , L1 : {} , Laplacian : {}, Sync : {}'.format(
                                    global_epoch, (sum(running_loss)/iter_in_batch), (sum(running_l1_loss)/iter_in_batch),
                                    (sum(runnig_laplacian_loss)/iter_in_batch), (sum(running_sync_loss)/iter_in_batch)
                                    ))


        print("Validation at Epochs {}".format(global_epoch))
        with torch.no_grad():

            vali_cur_loss, vali_cur_l1_loss, vali_cur_laplacian_loss, vali_cur_sync_loss, compare_fig = eval_generator(validation_dataset, device, generator, syncnet)
        
        # append logs 
        train_loss = np.append(train_loss, (sum(running_loss)/iter_in_batch))
        vali_loss = np.append(vali_loss, vali_cur_loss)

        # save logs
        save_logs(train_loss=train_loss, 
                  vali_loss=vali_loss,
                  model_name="generator",
                  savename='{}.csv'.format(save_name)
                  )

        
        # Tensorbard
        writer.add_scalar('Loss/train', sum(running_loss)/len(running_loss), global_epoch)
        writer.add_scalar('Loss/val', vali_cur_loss, global_epoch)
        writer.add_scalar('loss/train_sync', sum(running_loss)/iter_in_batch,global_epoch)
        writer.add_figure('Eval/compare', compare_fig, global_epoch)

        # save checkpoint
        save_checkpoint(generator, optimizer, checkpoint_dir, global_epoch)

        global_epoch +=1


"""
def save_checkpoint(model, checkpoint_dir,):

    checkpoint_path = join(
        checkpoint_dir, "model_epoch{}.pth".format(global_epoch))
    torch.save(model.state_dict(), checkpoint_path)
    print("Saved checkpoints:", checkpoint_path)
"""

def eval_generator(dataset, device, generator, syncnet):

    eval_l1_loss = []
    eval_laplacian_loss = []
    eval_sync_loss = []
    eval_total_loss = []
    eval_progress_bar = tqdm(enumerate(dataset))
    p = 0
    for _, (con_lip, seq_mels, mel, gt_lip) in eval_progress_bar:

        #  set generator in evaluation mode
        generator.eval()
        con_lip = con_lip.to(device)
        seq_mels = seq_mels.to(device)
        mel = mel.to(device)
        gt_lip = gt_lip.to(device)

        gen_lip, _ = generator(seq_mels, con_lip)

        gt_lip = gt_lip.reshape(gen_lip.shape[0], -1, 60)

        if p==0:
            com_fig = plot_compareLip(gen_lip, gt_lip)
        p +=1
        # L1 loss
        l1_loss = 0
        for idx in range(gen_lip.shape[1]):
            l1_loss += nn.functional.l1_loss(gen_lip[:, idx, :], gt_lip[:, idx, :])
        l1_loss = l1_loss/5
        laplacian_loss = 0
        sync_loss = get_sync_loss(mel, gen_lip, syncnet,p,True)
        loss = l1_loss #+ laplacian_loss + sync_loss

        eval_l1_loss.append(l1_loss.item())
        # eval_laplacian_loss.append(laplacian_loss.item())
        eval_sync_loss.append(sync_loss.item())
        eval_total_loss.append(loss.item())
        eval_progress_bar.set_description("Validation: Total Loss : {} , L1 : {} , Sync : {}".format(
                  sum(eval_total_loss)/len(eval_total_loss), sum(eval_l1_loss)/len(eval_l1_loss), sum(eval_sync_loss)/len(eval_sync_loss)))

    avg_l1_loss = sum(eval_l1_loss)/len(eval_l1_loss)
    avg_laplacian_loss = 0  # sum(eval_laplacian_loss)/len(eval_laplacian_loss)
    avg_sync_loss = sum(eval_sync_loss)/len(eval_sync_loss)
    avg_loss = sum(eval_total_loss)/len(eval_total_loss)

    return avg_loss, avg_l1_loss, avg_laplacian_loss, avg_sync_loss, com_fig


if __name__ == "__main__":

    save_name = args.save_name
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    ckpt_syncnet_path = args.checkpoint_syncnet_path
    batch_size = hparams.batch_size * torch.cuda.device_count()

    writer = SummaryWriter('../tensorboard/{}'.format(save_name))

    print("Running save name :  {}".format(save_name))

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)


    # Load train_dataset and
    train_dataset = Dataset(split='train', args = args)

    validation_dataset = Dataset(split='test', args = args)

    train_data_loader = dataset.DataLoader(train_dataset,
                                           batch_size=hparams.batch_size,
                                           shuffle=True,
                                           num_workers=hparams.num_workers)

    validation_data_loader = dataset.DataLoader(validation_dataset,
                                                batch_size=hparams.batch_size,
                                                shuffle=False,
                                                num_workers=hparams.num_workers)

    print("Training Dataset : {}".format(len(train_dataset)))
    print("Validation Dataset : {}".format(len(validation_dataset)))

    #** Model Section*** 
    # Landmark generator
    generator = Lip_Gen().to(device=device)
    # SyncNet
    syncnet = Syncnet().to(device=device)


    print("Loading SyncNet checkpoints ......")
    syncnet = load_checkpoint(path = ckpt_syncnet_path,
                    model = syncnet,
                    optimizer = None,
                    use_cuda = use_cuda,
                    reset_optimizer=True,
                    pretrain=True
                    )
    print("Finish loading SyncNet checkpoints")

    print("Trainable parameter in Generator : {}".format(sum(params.numel() for params in generator.parameters() if params.requires_grad)))
    # optimizer
    optimizer = optim.Adam([params for params in generator.parameters() if params.requires_grad], lr=0.0001)


    continue_ckpt = False
    if checkpoint_path is not None:

        continue_ckpt =True

        generator, optimizer, global_epoch = load_checkpoint(path = checkpoint_path,
                                                             model = generator,
                                                             optimizer = optimizer,
                                                             use_cuda = use_cuda, 
                                                             reset_optimizer = False,
                                                             pretain=False
                                                             )

    if torch.cuda.device_count() > 1:

        generator = DataParallel(generator)
        print("Training model parallel with {}".format(torch.cuda.device_count()))


    
    # set both model to device again and set syncnet to evaluation stage
    generator.to(device)
    syncnet.to(device)
    syncnet.eval()

    # computational graph visualization on tensorboard
    """
    dataiter = iter(train_data_loader)
    lip , mel,_ ,_= dataiter.next()
    writer.add_graph(generator, (mel.to(device),lip.to(device)))
    """

    print("Start training generator")
    train(device=device,
          generator=generator,
          syncnet=syncnet,
          train_dataset=train_data_loader,
          validation_dataset=validation_data_loader,
          optimizer=optimizer,
          save_name=save_name,
          checkpoint_dir=checkpoint_dir,
          nepochs=hparams.nepochs,
          continue_ckpt=continue_ckpt
          )
    print("Finish training Generator")

    writer.close()








