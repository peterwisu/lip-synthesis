import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch import optim
import numpy as np
from hparams import hparams 
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset.generator import Dataset
from torch.utils import data as data_utils
from utils.front import frontalize_landmarks
from src.models.lstmgen import LstmGen as Generator
from src.models.syncnet import SyncNet
from torch.utils.tensorboard import SummaryWriter
from utils.plot import plot_compareLip, plot_visLip, plot_comp, plot_seqlip_comp
from utils.wav2lip import load_checkpoint , save_checkpoint
from utils.utils import save_logs,load_logs 
from utils.loss  import CosineBCELoss




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class TrainEnd2End():
    """
    
    """
    
    
    def __init__ (self, args):
        
        
        # arguement and hyperparameters
        self.save_name  = args.save_name
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_path = args.checkpoint_path
        self.batch_size = hparams.e2e_batch_size
        self.global_epoch = 0
        self.nepochs = hparams.e2e_nepochs
        self.apply_disc = hparams.e2e_apply_disc
        self.sync_coeff = hparams.e2e_sync_coeff
        self.recon_coeff = hparams.e2e_recon_coeff
        self.gen_lr = hparams.e2e_gen_lr 
        self.disc_lr = hparams.e2e_disc_lr
        self.pretrain = args.pretrain_syncnet
        self.pretrain_path = args.pretrain_syncnet_path

        self.checkpoint_interval =  args.checkpoint_interval

           
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
        # load Syncnet 
        self.syncnet = SyncNet(end2end=True).to(device=device)
        
        #check if using pretrain Discriminator
        if self.pretrain:
            print("######################")
            print("Using Pretrain Syncnet")

            self.syncnet = load_checkpoint(path=self.pretrain_path,
                                           model=self.syncnet,
                                           optimizer=None,
                                           use_cuda=use_cuda,
                                           reset_optimizer=True,
                                           pretrain=True
                                           )
            self.syncnet.to(device)
            #self.syncnet.eval()

            
            
        else:

            print("###########################")
            print("Not using pretrain Syncnet")

            self.disc_optimizer = optim.SGD([params for params in self.syncnet.parameters() if params.requires_grad], lr=self.disc_lr)

        
        print("Finish loading Syncnet !!")



        """<----------------------------Generator------------------------------------------->""" 
         
        # load lip generator 
        self.generator  = Generator().to(device=device)
        
        self.gen_optimizer = optim.Adam([params for params in self.generator.parameters() if params.requires_grad], lr=self.gen_lr)

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
        

        """<----------List of loss funtion--------->"""
        # SyncLoss
        self.sync_loss = CosineBCELoss()
        # Mean  Square Error loss
        self.mse_loss = nn.MSELoss()
        # L1 loss 
        self.l1_loss = nn.L1Loss()
        # chosen reconstruction loss
        self.recon_loss = self.mse_loss

        


    def __get_disc_loss__ (self,disc_pred, y):

        """
        Calculate SyncLoss from Syncnet

        """

        # prdicted embedding
        s , v = disc_pred

        # create a torch tensor for the groundtruth
        y= torch.ones(s.shape[0],1).to(device) if y  == 1 else torch.zeros(s.shape[0],1).to(device)

        
        # caculate sync loss
        loss , acc = self.sync_loss(s,v,y) 
    
        return loss
 
        
    
    def __train_model__ (self):
        
        
        running_gen_disc_loss = 0.
        running_recon_loss = 0.
        running_disc_loss = 0.
        
        running_gen_loss =0
        iter_inbatch = 0
        
        prog_bar = tqdm(self.train_loader)
        
        for (con_fl, seq_mels, mel, gt_fl) in prog_bar:
            
            con_lip = con_fl[:,:,48:,:].to(device)
            con_face = con_fl[:,:,:48,:].to(device)
            gt_lip = gt_fl[:,:,48:,:].to(device)
            gt_face = gt_fl[:,:,:48,:].to(device)
            seq_mels = seq_mels.to(device)
            mel = mel.to(device)



            ###################### Discriminator #############################
            if  self.global_epoch >= self.apply_disc and not self.pretrain: 

                
                self.syncnet.train()
                self.disc_optimizer.zero_grad()
                # generate a fake lip from generator  
                fake_lip, _ =  self.generator(seq_mels, con_lip)
                disc_fake_pred = self.syncnet(mel,fake_lip.detach())
                disc_real_pred = self.syncnet(mel,gt_lip)
                
                
                disc_fake_loss = self.__get_disc_loss__(disc_fake_pred, y=0)
                disc_real_loss = self.__get_disc_loss__(disc_real_pred, y=1)

                disc_loss = disc_fake_loss + disc_real_loss
                disc_loss.backward(retain_graph=True)
                self.disc_optimizer.step()

            else : 

                disc_loss = torch.zeros(1)

            running_disc_loss += disc_loss.item()
            #################################################################


            
            ####################### Generator ###############################
            self.gen_optimizer.zero_grad()
            self.generator.train()
            gen_lip, _ = self.generator(seq_mels, con_lip)



            if self.global_epoch >=  self.apply_disc or self.pretrain:

                disc_gen_pred = self.syncnet(mel, gen_lip)
                gen_disc_loss = self.__get_disc_loss__(disc_gen_pred,y=1)

            else:

                gen_disc_loss = torch.zeros(1)
        
            gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
            gen_lip = gen_lip.reshape(gen_lip.size(0),-1) 
            recon_loss = self.recon_loss(gen_lip,gt_lip)          


            if self.global_epoch >=  self.apply_disc  or self.pretrain:
             
                gen_loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * gen_disc_loss)

            else :

                gen_loss = recon_loss

            gen_loss.backward()
            self.gen_optimizer.step()
            ####################################################################

        
            running_recon_loss += recon_loss.item() * self.recon_coeff if self.global_epoch >= self.apply_disc or self.pretrain else recon_loss.item()
            running_gen_disc_loss  += gen_disc_loss.item() * self.sync_coeff if self.global_epoch >= self.apply_disc  or self.pretrain else  gen_disc_loss.item()
            running_gen_loss += gen_loss.item()
            
            iter_inbatch+=1


             
            
            
            prog_bar.set_description("TRAIN Epochs: {} || Generator Loss : {:.5f} , Recon : {:.5f}, Sync : {:.5f} ||  Disciminator : {:.5f}".format(self.global_epoch,
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
                
                
                

                
                con_lip = con_fl[:,:,48:,:].to(device)
                con_face = con_fl[:,:,:48,:].to(device)
                gt_lip = gt_fl[:,:,48:,:].to(device)
                gt_face = gt_fl[:,:,:48,:].to(device)
                seq_mels = seq_mels.to(device)
                mel = mel.to(device)

                
                if self.global_epoch >=  self.apply_disc :
                ################### Discriminator ##########################

                    self.syncnet.eval()
                    # generate a fake lip from generator  
                    fake_lip, _ =  self.generator(seq_mels, con_lip)
                    disc_fake_pred = self.syncnet(mel,fake_lip.detach())
                    disc_real_pred = self.syncnet(mel,gt_lip)
                    
                    
                    disc_fake_loss = self.__get_disc_loss__(disc_fake_pred, y=0)
                    disc_real_loss = self.__get_disc_loss__(disc_real_pred, y=1)

                    disc_loss = disc_fake_loss + disc_real_loss 

                    running_disc_loss += disc_loss.item()

                else : 


                    disc_loss = torch.zeros(1)

                ################### Generator ##############################

                self.generator.eval()
                     
                gen_lip, _ = self.generator(seq_mels, con_lip)

                
                if self.global_epoch >=  self.apply_disc or self.pretrain:
                    disc_gen_pred = self.syncnet(mel, gen_lip)
                    gen_disc_loss = self.__get_disc_loss__(disc_gen_pred,y=1)

                else :

                    gen_disc_loss = torch.zeros(1)
            
                gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
                gen_lip = gen_lip.reshape(gen_lip.size(0),-1) 
                recon_loss = self.recon_loss(gen_lip,gt_lip)          
                ############################################################



                if self.global_epoch >=  self.apply_disc  or self.pretrain :

                    gen_loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * gen_disc_loss)

                else :

                    gen_loss = recon_loss



                running_recon_loss += recon_loss.item() * self.recon_coeff if self.global_epoch >= self.apply_disc  or self.pretrain else recon_loss.item()
                running_gen_disc_loss  += gen_disc_loss.item() * self.sync_coeff if self.global_epoch >= self.apply_disc or self.pretrain else  gen_disc_loss.item()
                running_gen_loss += gen_loss.item()

                iter_inbatch+=1
                
                
                prog_bar.set_description("VALI Epochs: {} || Generator Loss : {:.5f} , Recon : {:.5f}, Sync : {:.5f} || Disciminator : {:.5f}".format(self.global_epoch,
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


            if self.global_epoch % self.checkpoint_interval == 0:

                # save checkpoint
                save_checkpoint(self.generator, self.gen_optimizer, self.checkpoint_dir, self.global_epoch, '{}.pth'.format(self.save_name))
                self.__vis_vdo_result__()

            self.global_epoch +=1

    
    def start_training(self):

        print("Save name : {}".format(self.save_name))
        print("Using CUDA : {} ".format(use_cuda))
        
        if use_cuda: print ("Using {} GPU".format(torch.cuda.device_count())) 
        
        print("Training dataset {}".format(len(self.train_dataset)))
        print("Validation dataset {}".format(len(self.vali_dataset)))

        
        #self.__vis_comp_graph__()
        print("Start training END2END")

        self.__training_stage__()

        print("Finish Trainig END2END")

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

    def __vis_vdo_result__ (self):
        

        from .inference import Inference
        import argparse

        parser = argparse.ArgumentParser(description="File for running Inference")

        parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default='./checkpoints/end2end/{}.pth'.format(self.save_name))

        parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default='./checkpoints/image2image/image2image.pth',required=False)

        parser.add_argument('--input_face', type=str, help="File path for input videos/images contain face",default='dummy/me.jpeg', required=False)

        parser.add_argument('--input_audio', type=str, help="File path for input audio/speech as .wav files", default='./dummy/audio.mp3', required=False)

        parser.add_argument('--fps', type=float, help= "Can only be specified only if using static image default(25 FPS)", default=25,required=False)

        parser.add_argument('--fl_detector_batchsize',  type=int , help='Batch size for landmark detection', default = 32)

        parser.add_argument('--generator_batchsize', type=int, help="Batch size for Generator model", default=16)

        parser.add_argument('--output_name', type=str , help="Name and path of the output file", default="training_results.mp4")

        parser.add_argument('--vis_fl', type=bool, help="Visualize Facial Landmark ??", default=True)

        parser.add_argument('--test_img2img', type=bool, help="Testing image2image module with no lip generation" , default=False)

        args = parser.parse_args()

        eval_result = Inference(args=args)

        eval_result.start()




 
