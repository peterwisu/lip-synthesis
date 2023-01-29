import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch import optim
import numpy as np
from hparams import hparams
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from src.dataset.generator import Dataset
from torch.utils import data as data_utils
from utils.front import frontalize_landmarks
from src.models.lstmgen import LstmGen as Generator
from src.models.syncnet import SyncNet
from torch.utils.tensorboard import SummaryWriter
from utils.plot import plot_compareLip, plot_visLip, plot_comp, plot_seqlip_comp
from utils.wav2lip import load_checkpoint , save_checkpoint
from utils.utils import save_logs,load_logs 
from utils.loss import CosineBCELoss


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



class TrainGenerator():
    """
       
    """
    
    
    def __init__ (self, args):
        
        
        # arguement and hyperparameters
        self.save_name  = args.save_name
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_path = args.checkpoint_path
        self.ckpt_syncnet_path = args.checkpoint_syncnet_path
        self.batch_size = hparams.gen_batch_size
        self.apply_disc = hparams.gen_apply_disc
        self.global_epoch = 0
        self.nepochs = hparams.gen_nepochs
        self.lr =   hparams.gen_lr

        self.recon_coeff = hparams.gen_recon_coeff
        self.sync_coeff = hparams.gen_sync_coeff
        # frontalize weight
        #self.front_weight = np.load('./checkpoints/front/frontalization_weights.npy')

        
            
        # Tensorboard
        self.writer = SummaryWriter("../tensorboard/{}".format(self.save_name))

       
        
        """<---------------------------Dataset -------------------------------------->"""
        self.train_dataset = Dataset(split='train', args=args)
        
        self.vali_dataset = Dataset(split='val', args=args)
        
        self.train_loader = data_utils.DataLoader(self.train_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=hparams.num_workers)
        
        self.vali_loader = data_utils.DataLoader(self.vali_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=hparams.num_workers)
        
        


        """ <------------------------------SyncNet ------------------------------------->"""
        if self.ckpt_syncnet_path is not None:
            print("Loading SyncNet .......")  
            # load Syncnet 
            self.syncnet = SyncNet(pretrain=True,is3D=False).to(device=device)
            # load Syncnet checkpoint
            self.syncnet = load_checkpoint(path=self.ckpt_syncnet_path,
                                           model=self.syncnet,
                                           optimizer=None,
                                           use_cuda=use_cuda,
                                           reset_optimizer=True,
                                           pretrain=True)
            self.syncnet.to(device=device)
            self.syncnet.eval() 
               
            print("Finish loading Syncnet !!")

        else:

            print("No SyncNet ckpt provided, Not loading SyncNet from checkpoint")
            self.apply_disc = 100000000000000



        """<----------------------------Generator------------------------------------------->""" 
         
        # load lip generator 
        self.generator  = Generator().to(device=device)
        
        self.optimizer = optim.Adam([params for params in self.generator.parameters() if params.requires_grad], lr=self.lr)

        
        # load checkpoint if the path is given
        self.continue_ckpt = False
        if self.checkpoint_path is not None:

            self.continue_ckpt =True

            self.generator, self.optimizer, self.global_epoch = load_checkpoint(path = self.checkpoint_path,
                                                                model = self.generator,
                                                                optimizer = self.optimizer,
                                                                use_cuda = use_cuda, 
                                                                reset_optimizer = False,
                                                                pretain=False
                                                                )
            print("Load generator checkpoint")
            
            
        if self.continue_ckpt:
            
            self.train_loss , self.vali_loss = load_logs(model_name="generator", savename="{}.csv".format(self.save_name),epoch=self.global_epoch, type_model='generator')

            self.global_epoch +=1
        
        else:

            print("Not continue form Checkpoint")
            self.train_loss = np.array([])
            self.vali_loss = np.array([])
            
            
        """<-----------------------Parallel Trainining-------------------------------->""" 
        # If GPU detect more that one then train model in parallel 
        if torch.cuda.device_count() > 1:
             
            self.generator = DataParallel(self.generator)
            self.batch_size = self.batch_size * torch.cuda.device_count()
            print("Training or Testing model with  {} GPU " .format(torch.cuda.device_count()))
            
        self.generator.to(device) 
        

        """<----------List of reconstruction loss-------------------------------------->"""
        # Binary cross entrophy loss
        self.bce_loss = nn.BCELoss()
        # Mean  Square Error loss
        self.mse_loss = nn.MSELoss()
        # L1 loss 
        self.l1_loss = nn.L1Loss()
        # L1 smooth loss 
        self.l1_smooth = nn.SmoothL1Loss()
        

        # chosen reconstruction loss
        self.recon_loss = self.mse_loss
                            


        """ Evaluation video"""


        self.audio = None 

    def __get_sync_loss__ (self,audio, gen_lip):

        """
        Calculate SyncLoss from Syncnet
        """
        self.syncnet.eval()
        
        s, v = self.syncnet(audio, gen_lip)

        y = torch.ones(gen_lip.shape[0],1).to(device)
        
        loss , _ = CosineBCELoss(s,v,y) 
        
        return loss
        
    
    def __train_model__ (self):
        
        
        running_sync_loss = 0.
        running_recon_loss = 0.
        running_loss =0
        iter_inbatch = 0
        
        prog_bar = tqdm(self.train_loader)
        
        for (con_fl, seq_mels, mel, gt_fl) in prog_bar:

            
            self.optimizer.zero_grad()
            self.generator.train()
            
            con_lip = con_fl[:,:,48:,:].to(device)
            con_face = con_fl[:,:,:48,:].to(device)
            gt_lip = gt_fl[:,:,48:,:].to(device)
            gt_face = gt_fl[:,:,:48,:].to(device)


            seq_mels = seq_mels.to(device)
            mel = mel.to(device)
           
            gen_lip, _ = self.generator(seq_mels, con_lip) 

            #gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
            #gen_lip = gen_lip.reshape(gen_lip.size(0),-1)


            
            recon_loss = self.recon_loss(gt_lip.reshape(gt_lip.size(0),-1),gen_lip.reshape(gen_lip.size(0),-1))

                
            gen_lip = gen_lip[:,:,:40]

            sync_loss = self.__get_sync_loss__ (mel, gen_lip) if  self.global_epoch >=self.apply_disc  else  torch.zeros(1)
            
            
            loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * sync_loss)if  self.global_epoch >= self.apply_disc else recon_loss

            loss.backward()
            self.optimizer.step()


            running_recon_loss += recon_loss.item() * self.recon_coeff if self.global_epoch >= self.apply_disc else recon_loss.item()
            running_sync_loss  += sync_loss.item() * self.sync_coeff if self.global_epoch >= self.apply_disc else sync_loss.item()

            running_loss += loss.item()
            
            iter_inbatch+=1

            
            
            prog_bar.set_description("TRAIN Epochs: {} , Loss : {:.3f} , Recon : {:.3f}, Sync : {:.3f}".format(self.global_epoch,
                                                                                             running_loss/iter_inbatch,
                                                                                             running_recon_loss/iter_inbatch,
                                                                                             running_sync_loss/iter_inbatch))

        avg_loss =  running_loss / iter_inbatch
        avg_recon_loss = running_loss / iter_inbatch
        avg_sync_loss = running_sync_loss/ iter_inbatch

        return avg_loss, avg_recon_loss, avg_sync_loss


    def __eval_model__ (self, split=None):


        running_loss = 0.
        running_recon_loss =0.
        running_sync_loss =0.
        iter_inbatch = 0


        prog_bar = tqdm(self.vali_loader)

        with torch.no_grad():

            for (con_fl,seq_mels, mel, gt_fl) in prog_bar:

                
                self.generator.eval()
                con_lip = con_fl[:,:,48:,:].to(device)
                con_face = con_fl[:,:,:48,:].to(device)
                gt_lip = gt_fl[:,:,48:,:].to(device)
                gt_face = gt_fl[:,:,:48,:].to(device)


                seq_mels = seq_mels.to(device)
                mel = mel.to(device)
     

                gen_lip , _ = self.generator(seq_mels, con_lip)
            

                gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
                gen_lip = gen_lip.reshape(gen_lip.size(0),-1)
                
                recon_loss = self.recon_loss(gen_lip,gt_lip)
                 
            
                sync_loss = self.__get_sync_loss__ (mel, gen_lip) if  self.global_epoch >= self.apply_disc  else  torch.zeros(1)

                loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * sync_loss)if  self.global_epoch >= self.apply_disc else recon_loss
                
                running_recon_loss += recon_loss.item() * self.recon_coeff if self.global_epoch >= self.apply_disc else recon_loss.item()
                running_sync_loss  += sync_loss.item() * self.sync_coeff if self.global_epoch >= self.apply_disc else sync_loss.item()

                running_loss += loss.item() 
                iter_inbatch +=1


                prog_bar.set_description("VALI Epochs: {} , Loss : {:.3f} , Recon : {:.3f} , Sync : {:.3f}".format(self.global_epoch,
                                                                                             running_loss/iter_inbatch,
                                                                                             running_recon_loss/iter_inbatch,
                                                                                             running_sync_loss/iter_inbatch))

        avg_loss = running_loss /iter_inbatch
        avg_recon_loss = running_recon_loss/ iter_inbatch
        avg_sync_loss = running_sync_loss/ iter_inbatch


        return avg_loss, avg_recon_loss, avg_sync_loss


    def __update_logs__ (self, cur_train_loss, cur_vali_loss, cur_train_recon_loss,  cur_vali_recon_loss, cur_train_sync_loss, cur_vali_sync_loss, com_fig, com_seq_fig):


        self.train_loss = np.append(self.train_loss, cur_train_loss)
        self.vali_loss  = np.append(self.vali_loss, cur_vali_loss)

        save_logs(train_loss=self.train_loss, 
                  vali_loss=self.vali_loss,
                  model_name="generator",
                  savename='{}.csv'.format(self.save_name)
                  )

        # ******* plot metrics *********
        # plot metrics comparison (train vs validation)
        loss_comp  = plot_comp(self.train_loss,self.vali_loss,  name="Loss")
        # ***** Tensorboard ****
        # Figure
        self.writer.add_figure('Comp/loss', loss_comp, self.global_epoch)
        self.writer.add_scalar("Loss/train", cur_train_loss, self.global_epoch)
        self.writer.add_scalar("Loss/val" ,  cur_vali_loss ,self.global_epoch)
        
        self.writer.add_scalar("Loss/train_recon" ,  cur_train_recon_loss ,self.global_epoch)
        self.writer.add_scalar("Loss/val_recon" ,  cur_vali_recon_loss ,self.global_epoch)
        self.writer.add_scalar("Loss/train_sync" ,  cur_train_sync_loss ,self.global_epoch)
        self.writer.add_scalar("Loss/val_sync" ,  cur_vali_sync_loss ,self.global_epoch)
        self.writer.add_figure("Vis/compare", com_fig, self.global_epoch)


        for frame in range(len(com_seq_fig)):

            self.writer.add_figure("Vis/seq", com_seq_fig[frame], frame)



        
    def __training_stage__ (self):
        

        while self.global_epoch < self.nepochs:



            cur_train_loss , cur_train_recon_loss , cur_train_sync_loss= self.__train_model__() 

            cur_vali_loss , cur_vali_recon_loss  ,  cur_vali_sync_loss= self.__eval_model__()

            com_fig = self.__compare_lip__()

            com_seq_fig = self.__vis_seq_result__()

            self.__update_logs__(cur_train_loss, cur_vali_loss,  cur_train_recon_loss,  cur_vali_recon_loss, cur_train_sync_loss, cur_vali_sync_loss, com_fig, com_seq_fig)

            # save checkpoint
            save_checkpoint(self.generator, self.optimizer, self.checkpoint_dir, self.global_epoch, self.save_name)

            self.global_epoch +=1

    
    def start_training(self):

        print("Save name : {}".format(self.save_name))
        print("Using CUDA : {} ".format(use_cuda))
        
        if use_cuda: print ("Using {} GPU".format(torch.cuda.device_count())) 
        
        print("Training dataset {}".format(len(self.train_dataset)))
        print("Validation dataset {}".format(len(self.vali_dataset)))

        
        self.__vis_comp_graph__()
        print("Start training SyncNet")

        self.__training_stage__()

        print("Finish Trainig SyncNet")

        self.writer.close()


    def __vis_comp_graph__ (self):

        data = iter(self.vali_loader)

        (con_fl, seq_mels, _, _ ) = next(data)
        
        # take only lip
        con_lip  =  con_fl[:,:,48:,:] 

        # computational graph visualization on tensorboard
        self.writer.add_graph(self.generator, (seq_mels.to(device),con_lip.to(device)))

        del data 
        


    def __compare_lip__ (self):

        # iterate over validation loader        
        data = iter(self.vali_loader)
        
        # retrive the data in firsr iteration
        (con_fl, seq_mels, mel, gt_fl) = next(data)
        
        with torch.no_grad():

            self.generator.eval() 
            con_lip = con_fl[:,:,48:,:].to(device)
            con_face = con_fl[:,:,:48,:].to(device)
            gt_lip = gt_fl[:,:,48:,:].to(device)
            gt_face = gt_fl[:,:,:48,:].to(device)


            seq_mels = seq_mels.to(device)
            mel = mel.to(device)
            gen_lip , _ = self.generator(seq_mels, con_lip)
 
            gen,ref, gt = gen_lip[0].detach().clone().cpu().numpy() ,con_lip[0].detach().clone().cpu().numpy(), gt_lip[0].detach().clone().cpu().numpy()
            gen = gen.reshape(5,20,-1)
            ref = ref.reshape(5,20,-1)
            gt = gt.reshape(5,20,-1)
            
            
            # plot compare figure
            com_fig = plot_seqlip_comp(gen[0],ref[0],gt[0])
        
        return com_fig
    
    def __vis_seq_result__ (self):

        data = iter(self.vali_loader)

        (con_fl, seq_mels , mel , gt_fl)  = next(data)
        
        # take only first sample from first batch
        #con_lip , seq_mels , mel , gt_lip  =  con_lip[0] , seq_mels[0] , mel[0] , gt_lip[0]

        with torch.no_grad() : 

            self.generator.eval()

            con_lip  = con_fl[:,:,48:,:].to(device)
            con_face = con_fl[:,:,:48,:].to(device)
            gt_lip   = gt_fl[:,:,48:,:].to(device)
            gt_face  = gt_fl[:,:,:48,:].to(device)

            seq_mels = seq_mels.to(device)
            mel = mel.to(device)
    
            gen_lip, _ = self.generator(seq_mels, con_lip)
            
        
            gen,ref, gt = gen_lip[0].detach().clone().cpu().numpy() ,con_lip[0].detach().clone().cpu().numpy(), gt_lip[0].detach().clone().cpu().numpy()
            
            gen = gen.reshape(5,20,-1)
            ref = ref.reshape(5,20,-1)
            gt = gt.reshape(5,20,-1)
            
            seq_fig = []
            for idx in range(5):
                
                
                vis_fig = plot_seqlip_comp(gen[idx],ref[idx],gt[idx])

                seq_fig.append(vis_fig)
    
        
        return seq_fig





        






            
            
    
            
        
        
        
        
        
        
        
        
        
