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

from src.dataset.end2end import Dataset
from torch.utils import data as data_utils
from utils.front import frontalize_landmarks

from src.models.generator import LstmGen as Generator
#from src.models.cus_gen import  Generator
#from src.models.fl_syncnet import SyncNet_fl
from src.models.end2end_sync import  SyncNet


from torch.utils.tensorboard import SummaryWriter
from utils.plot import plot_compareLip, plot_visLip, plot_comp, plot_seqlip_comp
from utils.wav2lip import load_checkpoint , save_checkpoint
from utils.utils import save_logs,load_logs , norm_lip2d

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class End2End():
    """
    
    """
    
    
    def __init__ (self, args):
        
        
        # arguement and hyperparameters
        self.save_name  = args.save_name
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_path = args.checkpoint_path
        self.batch_size = hparams.batch_size
        self.global_epoch = 0
        self.nepochs = hparams.nepochs
        self.apply_disc = args.apply_disc
        
        # if create checkpoint dir if it does not exist
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
            
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
        
        


        """ <------------------------------SyncNet Discriminator ------------------------------------->"""
        print("Loading SyncNet .......")  
        # load Syncnet 
        self.syncnet = SyncNet().to(device=device)

        self.disc_optimizer = optim.SGD([params for params in self.syncnet.parameters() if params.requires_grad], lr=0.0001)
      
        # frontalize weight
        self.front_weight = np.load('./checkpoints/front/frontalization_weights.npy')

        
        print("Finish loading Syncnet !!")



        """<----------------------------Generator------------------------------------------->""" 
         
        # load lip generator 
        self.generator  = Generator().to(device=device)
        
        self.gen_optimizer = optim.Adam([params for params in self.generator.parameters() if params.requires_grad], lr=0.0001)

        # load checkpoint if the path is given
        self.continue_ckpt = False
        if self.checkpoint_path is not None:

            self.continue_ckpt =True

            self.generator, self.gen_optimizer, self.global_epoch = load_checkpoint(path = self.checkpoint_path,
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
        self.recon_loss = self.l1_smooth

        self.sync_coeff = 0.3

        self.recon_coeff =0.7


    def __frontal_lip__(self,gen_lip,gt_face):

        
        #  Shape (B, 5, 20, 2)
        gen_lip = gen_lip.detach().clone().cpu().numpy().reshape(gen_lip.size(0),gt_face.size(1),-1,gt_face.size(3))

        #  Shape (B, 5, 48, 2)
        gt_face = gt_face.detach().clone().cpu().numpy()
    
        fl = np.concatenate((gt_face,gen_lip), axis=2)
        
        # list of frontal fl
        frontal_batch = []
        for b_idx in range(fl.shape[0]):
            frontal_seq = [] 
            for s_idx in range(fl.shape[1]):
                
                try:
                    frontal, _ = frontalize_landmarks(fl[b_idx, s_idx], self.front_weight) 

                except Exception as e : 

                    print(e)
                    frontal = fl[b_idx,s_idx]
                    #self.__plot_lip__(frontal)
                #self.__plot_lip__(frontal)
                frontal_seq.append(frontal)

            frontal_seq = np.stack(frontal_seq, axis=0) # (5, 68,2)
            frontal_batch.append(frontal_seq)

        frontal_batch = np.stack(frontal_batch, axis=0) #(B,5,68,2)

        frontal_lip = frontal_batch[:,:,48:,:] 
        
        #self.__plot_lip__(vis[:48])
        #vis_front = frontalize_landmarks(vis,self.front_weight)
        #self.__plot_lip__(vis_front)


        return frontal_lip

    def __plot_lip__(self,fl):


        plt.scatter(fl[:,0],fl[:,1])
        plt.plot(fl[0:7,0],fl[0:7,1], c="tab:pink", linewidth=3 )
        plt.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:pink", linewidth=3 )
        plt.plot(fl[12:17,0],fl[12:17,1], c="tab:pink", linewidth=3 )
        plt.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:pink", linewidth=3)
        #if line : plt.axvline(x = 1, color = 'b', linestyle='--',label = 'axvline - full height')
    
        plt.show()



    def __lip_normalization__(self,fl_lip):

        
        norm_batch = []

        for b_idx in range(fl_lip.shape[0]):

            norm_seqs = []

            for s_idx in range(fl_lip.shape[1]):

                if s_idx == 0: 

                    norm_lip, norm_distance = norm_lip2d(fl_lip[b_idx,s_idx])

                else:

                    norm_lip, _ = norm_lip2d(fl_lip[b_idx,s_idx], norm_distance)

                norm_seqs.append(norm_lip)

            norm_seqs = np.stack(norm_seqs, axis=0)

            norm_batch.append(norm_seqs)

        norm_batch = np.stack(norm_batch, axis=0)

        return norm_batch
 
    def __get_sync_loss__ (self,audio, gen_lip,gt_face):

        """
        Calculate SyncLoss from Syncnet
        """
        
        # frontalize a lip landmark

        # Get a tensor that share same storage, but does not track history
        # So that the frontalization and normalization wont affect gradient of model outputs
        gen_lip_data = gen_lip.data

        lip = self.__frontal_lip__(gen_lip_data, gt_face)
        lip = self.__lip_normalization__(lip)
        lip = lip.reshape(lip.shape[0],-1) 

        # copy value from numpy array to tensor in tensor that shar storage
        gen_lip_data.copy_(torch.from_numpy(lip)) 

        """  
        Code above would replace a value of gen_lip with frontalize and normalize lip 
        with out affecting the gradient and computation graph of model by replace a value 
        of variable of gen_lip_data which its storage is share with gen_lip 
        """

        gen_lip = gen_lip.reshape(gen_lip.size(0),5,-1) #  reshape tp satisfy the input shape of SyncNet
        
    
        self.syncnet.eval()

         
        with torch.backends.cudnn.flags(enabled=False):

            audio_emb, fl_emb = self.syncnet(audio, gen_lip) 
        
            # calculate similarity score
            sim_score = nn.functional.cosine_similarity(audio_emb, fl_emb)

            # create a torch tensor for the groundtruth
            # generate the tensor of value ones since
            # we want generated lip to sync to the speech 
            y = torch.ones(gen_lip.shape[0], 1).to(device)
            
            loss = self.bce_loss(sim_score.unsqueeze(1), y )
        
        return loss

    def __get_disc_loss__ (self,disc_pred, y):

        """
        Calculate SyncLoss from Syncnet

        """

        audio_emb , fl_emb = disc_pred


        # calculate similarity score
        sim_score = nn.functional.cosine_similarity(audio_emb, fl_emb)

        sim_score = sim_score.unsqueeze(1)

        # create a torch tensor for the groundtruth
        # generate the tensor of value ones since
        # we want generated lip to sync to the speech 
        y_tensor = torch.ones_like(sim_score) if y  == 1 else torch.zeros_like(sim_score)

        
        loss = self.bce_loss(sim_score, y_tensor)
    
        return loss
 
        
    
    def __train_model__ (self):
        
        
        running_gen_disc_loss = 0.
        running_recon_loss = 0.
        running_disc_loss = 0.
        
        running_gen_loss =0
        iter_inbatch = 0
        
        prog_bar = tqdm(self.train_loader)
        
        for (con_fl, seq_mels, mel, gt_fl) in prog_bar:
            
            #self.optimizer.zero_grad()
            self.generator.train()
            self.syncnet.train()
            
            con_lip = con_fl[:,:,48:,:].to(device)
            con_face = con_fl[:,:,:48,:].to(device)
            gt_lip = gt_fl[:,:,48:,:].to(device)
            gt_face = gt_fl[:,:,:48,:].to(device)
            seq_mels = seq_mels.to(device)
            mel = mel.to(device)



            if  self.global_epoch >= self.apply_disc : 
                ## Discriminator ## 
                
                self.disc_optimizer.zero_grad()
                # generate a fake lip from generator  
                fake_lip, _ =  self.generator(seq_mels, con_lip)
                disc_fake_pred = self.syncnet(mel,fake_lip.detach())
                disc_shuffle_pred = self.syncnet(mel,con_lip)
                #self.__plot_lip__(gt_lip[0,0].detach().clone().cpu().numpy())
                disc_real_pred = self.syncnet(mel,gt_lip)
                
                
                disc_fake_loss = self.__get_disc_loss__(disc_fake_pred, y=0)
                disc_shuffle_loss = self.__get_disc_loss__(disc_shuffle_pred, y=0)
                disc_real_loss = self.__get_disc_loss__(disc_real_pred, y=1)

                disc_loss = disc_fake_loss + disc_real_loss + disc_shuffle_loss
                disc_loss.backward(retain_graph=True)
                self.disc_optimizer.step()

            else : 

                disc_loss = torch.zeros(1)

            running_disc_loss += disc_loss.item()


            
            ##  Generator ## 
            self.gen_optimizer.zero_grad()
            gen_lip, _ = self.generator(seq_mels, con_lip)



            if self.global_epoch >=  self.apply_disc :

                disc_gen_pred = self.syncnet(mel, gen_lip)
                gen_disc_loss = self.__get_disc_loss__(disc_gen_pred,y=1)

            else:

                gen_disc_loss = torch.zeros(1)
        
            gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
            gen_lip = gen_lip.reshape(gen_lip.size(0),-1) 
            recon_loss = self.recon_loss(gen_lip,gt_lip)          


            if self.global_epoch >=  self.apply_disc :
             
                gen_loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * gen_disc_loss)

            else :

                gen_loss = recon_loss

            gen_loss.backward()
            self.gen_optimizer.step()

        
            running_recon_loss += recon_loss.item()
            running_gen_disc_loss  += gen_disc_loss.item()
            running_gen_loss += gen_loss.item()
            
            iter_inbatch+=1


             
            
            
            prog_bar.set_description("TRAIN Epochs: {} , Generator Loss : {:.3f} , Recon : {:.3f}, Sync : {:.3f}, Disciminator : {:.3f}".format(self.global_epoch,
                                                                                             running_gen_loss/iter_inbatch,
                                                                                             running_recon_loss/iter_inbatch,
                                                                                             running_gen_disc_loss/iter_inbatch,
                                                                                             running_disc_loss/iter_inbatch))

        avg_gen_loss =  running_gen_loss / iter_inbatch
        avg_recon_loss = running_recon_loss / iter_inbatch
        avg_gen_disc_loss = running_gen_disc_loss/ iter_inbatch
        avg_disc_loss = running_disc_loss/iter_inbatch

        return avg_gen_loss, avg_recon_loss, avg_gen_disc_loss, avg_disc_loss


    def __eval_model__ (self, split=None):

        running_gen_disc_loss = 0.
        running_recon_loss = 0.
        running_disc_loss = 0.
        
        running_gen_loss =0
        iter_inbatch = 0
        
        prog_bar = tqdm(self.vali_loader)
        
        with torch.no_grad():
            for (con_fl, seq_mels, mel, gt_fl) in prog_bar:
                
                
                #self.optimizer.zero_grad()
                self.generator.eval()
                self.syncnet.eval()
                
                con_lip = con_fl[:,:,48:,:].to(device)
                con_face = con_fl[:,:,:48,:].to(device)
                gt_lip = gt_fl[:,:,48:,:].to(device)
                gt_face = gt_fl[:,:,:48,:].to(device)
                seq_mels = seq_mels.to(device)
                mel = mel.to(device)

                
                if self.global_epoch >=  self.apply_disc :
                ## Discriminator ## 

                    # generate a fake lip from generator  
                    fake_lip, _ =  self.generator(seq_mels, con_lip)
                    disc_fake_pred = self.syncnet(mel,fake_lip.detach())
                    disc_shuffle_pred = self.syncnet(mel,con_lip)
                    #self.__plot_lip__(gt_lip[0,0].detach().clone().cpu().numpy())
                    disc_real_pred = self.syncnet(mel,gt_lip)
                    
                    
                    disc_fake_loss = self.__get_disc_loss__(disc_fake_pred, y=0)
                    disc_shuffle_loss = self.__get_disc_loss__(disc_shuffle_pred, y=0)
                    disc_real_loss = self.__get_disc_loss__(disc_real_pred, y=1)

                    disc_loss = disc_fake_loss + disc_real_loss + disc_shuffle_loss

                    running_disc_loss += disc_loss.item()

                else : 


                    disc_loss = torch.zeros(1)
                     
                gen_lip, _ = self.generator(seq_mels, con_lip)

                
                if self.global_epoch >=  self.apply_disc :
                    disc_gen_pred = self.syncnet(mel, gen_lip)
                    gen_disc_loss = self.__get_disc_loss__(disc_gen_pred,y=1)

                else :

                    gen_disc_loss = torch.zeros(1)
            
                gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
                gen_lip = gen_lip.reshape(gen_lip.size(0),-1) 
                recon_loss = self.recon_loss(gen_lip,gt_lip)          



                if self.global_epoch >=  self.apply_disc :

                    gen_loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * gen_disc_loss)

                else :

                    gen_loss = recon_loss


                running_recon_loss += recon_loss.item()
                running_gen_disc_loss  += gen_disc_loss.item()
                running_gen_loss += gen_loss.item()
                
                iter_inbatch+=1
                
                
                prog_bar.set_description("VALI Epochs: {} , Generator Loss : {:.3f} , Recon : {:.3f}, Sync : {:.3f}, Disciminator : {:.3f}".format(self.global_epoch,
                                                                                                 running_gen_loss/iter_inbatch,
                                                                                                 running_recon_loss/iter_inbatch,
                                                                                                 running_gen_disc_loss/iter_inbatch,
                                                                                                 running_disc_loss/iter_inbatch))

            avg_gen_loss =  running_gen_loss / iter_inbatch
            avg_recon_loss = running_recon_loss / iter_inbatch
            avg_gen_disc_loss = running_gen_disc_loss/ iter_inbatch
            avg_disc_loss = running_disc_loss/iter_inbatch


        return avg_gen_loss, avg_recon_loss, avg_gen_disc_loss, avg_disc_loss
    


    def __update_logs__ (self,
                         cur_train_gen_loss,
                         cur_vali_gen_loss,
                         cur_train_recon_loss,
                         cur_vali_recon_loss,
                         cur_train_sync_loss,
                         cur_vali_sync_loss,
                         cur_train_disc_loss,
                         cur_vali_disc_loss,
                         com_fig, com_seq_fig):


       # self.train_loss = np.append(self.train_loss, cur_train_loss)
       # self.vali_loss  = np.append(self.vali_loss, cur_vali_loss)

       # save_logs(train_loss=self.train_loss, 
       #           vali_loss=self.vali_loss,
       #           model_name="generator",
       #           savename='{}.csv'.format(self.save_name)
       #           )

        # ******* plot metrics *********
        # plot metrics comparison (train vs validation)
        loss_comp  = plot_comp(self.train_loss,self.vali_loss,  name="Loss")
        # ***** Tensorboard ****
        # Figure
        self.writer.add_figure('Comp/loss', loss_comp, self.global_epoch)
        self.writer.add_scalar("Loss/train_gen", cur_train_gen_loss, self.global_epoch)
        self.writer.add_scalar("Loss/val_gen" ,  cur_vali_gen_loss ,self.global_epoch)

        self.writer.add_scalar("Loss/train_disc", cur_train_disc_loss, self.global_epoch)
        self.writer.add_scalar("Loss/val_disc" ,  cur_vali_disc_loss ,self.global_epoch)
        
        self.writer.add_scalar("Loss/train_recon" ,  cur_train_recon_loss ,self.global_epoch)
        self.writer.add_scalar("Loss/val_recon" ,  cur_vali_recon_loss ,self.global_epoch)
        self.writer.add_scalar("Loss/train_sync" ,  cur_train_sync_loss ,self.global_epoch)
        self.writer.add_scalar("Loss/val_sync" ,  cur_vali_sync_loss ,self.global_epoch)
        self.writer.add_figure("Vis/compare", com_fig, self.global_epoch)


        for frame in range(len(com_seq_fig)):

            self.writer.add_figure("Vis/seq", com_seq_fig[frame], frame)



        
    def __training_stage__ (self):
        

        while self.global_epoch < self.nepochs:



            cur_train_gen_loss , cur_train_recon_loss , cur_train_sync_loss, cur_train_disc_loss= self.__train_model__() 

            cur_vali_gen_loss , cur_vali_recon_loss  ,  cur_vali_sync_loss, cur_vali_disc_loss = self.__eval_model__()

            com_fig = self.__compare_lip__()

            com_seq_fig = self.__vis_seq_result__()

            self.__update_logs__(cur_train_gen_loss, cur_vali_gen_loss,  cur_train_recon_loss,  cur_vali_recon_loss, cur_train_sync_loss, cur_vali_sync_loss, cur_train_disc_loss, cur_vali_disc_loss, com_fig, com_seq_fig)

            # save checkpoint
            #save_checkpoint(self.generator, self.optimizer, self.checkpoint_dir, self.global_epoch)

            self.global_epoch +=1

    
    def start_training(self):

        print("Save name : {}".format(self.save_name))
        print("Using CUDA : {} ".format(use_cuda))
        
        if use_cuda: print ("Using {} GPU".format(torch.cuda.device_count())) 
        
        print("Training dataset {}".format(len(self.train_dataset)))
        print("Validation dataset {}".format(len(self.vali_dataset)))

        
        #self.__vis_comp_graph__()
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

 
