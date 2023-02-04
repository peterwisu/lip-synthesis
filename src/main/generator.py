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
from src.models.lstmgen import LstmGen as Generator
from src.models.syncnet import SyncNet
from torch.utils.tensorboard import SummaryWriter
from utils.plot import  plot_comp, plot_lip_comparision
from utils.wav2lip import load_checkpoint , save_checkpoint
from utils.utils import save_logs,load_logs 
from utils.loss  import CosineBCELoss


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class TrainGenerator():
    """
    ************************
    Training Generator Model
    ************************
    """
    def __init__ (self, args):
        # arguement and hyperparameters
        self.save_name  = args.save_name
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_path = args.checkpoint_path
        self.batch_size = hparams.gen_batch_size
        self.global_epoch = 0
        self.nepochs = hparams.gen_nepochs
        self.sync_coeff = hparams.gen_sync_coeff
        self.recon_coeff = hparams.gen_recon_coeff
        self.gen_lr = hparams.gen_gen_lr 
        self.disc_lr = hparams.gen_disc_lr
        self.train_type = args.train_type
        self.pretrain_path = args.pretrain_syncnet_path
        
        # if not using Discriminator
        if self.train_type == "gen":

            self.recon_coeff = 1.0
            self.sync_coeff  = 0 



        self.checkpoint_interval =  args.checkpoint_interval

        if (self.train_type != "gen") and (self.recon_coeff + self.sync_coeff) != 1 :

            raise ValueError("Sum of the loss coeff should be sum up to 1, the recon_coeff is {} and sync_coeff is {}".format(self.recon_coeff,self.sync_coeff))

           
        # Tensorboard for log and result visualization
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
        if self.train_type != "gen":
            # load Syncnet model
            self.syncnet = SyncNet().to(device=device)
            
            #check if using pretrain Discriminator
            if self.train_type == "pretrain":
                print("######################")
                print("Using Pretrain Syncnet")
                print("######################")

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

                print("##################################################################")
                print("Not using pretrain Syncnet and training it together with generator")
                print("##################################################################")

                self.disc_optimizer = optim.Adam([params for params in self.syncnet.parameters() if params.requires_grad], lr=self.disc_lr)

            
            print("Finish loading Syncnet !!")
        else: 
            
            print("##############################################################")
            print("Not using Discriminator(SyncNet), training only generator")
            print("##############################################################")



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
        loss , _, _,_ = self.sync_loss(s,v,y) 
    
        return loss
 
        
    
    def __train_model__ (self):
        
        
        running_gen_disc_loss = 0.
        running_recon_loss = 0.
        running_disc_loss = 0.
        
        running_gen_loss =0
        iter_inbatch = 0
        
        prog_bar = tqdm(self.train_loader)
        
        for (con_fl, seq_mels, mel, gt_fl) in prog_bar:
            
            con_lip = con_fl.to(device)
            gt_lip = gt_fl.to(device)
            seq_mels = seq_mels.to(device)
            mel = mel.to(device)



            ###################### Discriminator #############################
            if  self.train_type == "end2end": 

                
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


            if  self.train_type != "gen":
            
                disc_gen_pred = self.syncnet(mel, gen_lip)
                gen_disc_loss = self.__get_disc_loss__(disc_gen_pred,y=1)

            else:

                gen_disc_loss = torch.zeros(1)
        
            gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
            gen_lip = gen_lip.reshape(gen_lip.size(0),-1) 
            recon_loss = self.recon_loss(gen_lip,gt_lip)          


            if  self.train_type != "gen":
             
                gen_loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * gen_disc_loss)

            else :

                gen_loss = recon_loss

            gen_loss.backward()
            self.gen_optimizer.step()
            ####################################################################

        
            running_recon_loss += recon_loss.item() * self.recon_coeff if self.train_type != "gen" else recon_loss.item()

            running_gen_disc_loss  += gen_disc_loss.item() * self.sync_coeff if  self.train_type != "gen" else  gen_disc_loss.item()
            running_gen_loss += gen_loss.item()
            
            iter_inbatch+=1


             
            
            
            prog_bar.set_description("TRAIN Epochs: {} || Generator Loss : {:.5f} , Recon({})({}) : {:.5f}, Sync({}) : {:.5f} || Disciminator : {:.5f}".format(
                                                                                             self.global_epoch,
                                                                                             running_gen_loss/iter_inbatch,
                                                                                             self.recon_coeff,
                                                                                             self.recon_loss,
                                                                                             running_recon_loss/iter_inbatch,
                                                                                             self.sync_coeff,
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
                
                con_lip = con_fl.to(device)
                gt_lip = gt_fl.to(device)
                seq_mels = seq_mels.to(device)
                mel = mel.to(device)


                if  self.train_type == "end2end": 
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

                
                if  self.train_type != "gen":
                    disc_gen_pred = self.syncnet(mel, gen_lip)
                    gen_disc_loss = self.__get_disc_loss__(disc_gen_pred,y=1)

                else :

                    gen_disc_loss = torch.zeros(1)
            
                gt_lip = gt_lip.reshape(gt_lip.size(0),-1)
                gen_lip = gen_lip.reshape(gen_lip.size(0),-1) 
                recon_loss = self.recon_loss(gen_lip,gt_lip)          
                ############################################################



                if  self.train_type != "gen":

                    gen_loss  = (self.recon_coeff * recon_loss) + (self.sync_coeff * gen_disc_loss)

                else :

                    gen_loss = recon_loss



                running_recon_loss += recon_loss.item() * self.recon_coeff if self.train_type != "gen" else recon_loss.item()
                running_gen_disc_loss  += gen_disc_loss.item() * self.sync_coeff if self.train_type != "gen" else  gen_disc_loss.item()
                running_gen_loss += gen_loss.item()

                iter_inbatch+=1
                
                
                prog_bar.set_description("VALI Epochs : {} || Generator Loss : {:.5f} , Recon({})({}) : {:.5f}, Sync({}) : {:.5f} || Disciminator : {:.5f}".format(
                                                                                                 self.global_epoch,
                                                                                                 running_gen_loss/iter_inbatch,
                                                                                                 self.recon_coeff,
                                                                                                 self.recon_loss,
                                                                                                 running_recon_loss/iter_inbatch,
                                                                                                 self.sync_coeff,
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


       #self.train_loss = np.append(self.train_loss, cur_train_loss)
       #self.vali_loss  = np.append(self.vali_loss, cur_vali_loss)

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

            com_fig, com_seq_fig = self.__vis_lip_result__()

            self.__update_logs__(cur_train_gen_loss, cur_vali_gen_loss,  cur_train_recon_loss,  cur_vali_recon_loss, cur_train_sync_loss, cur_vali_sync_loss, cur_train_disc_loss, cur_vali_disc_loss, com_fig, com_seq_fig)


            if (((self.global_epoch % self.checkpoint_interval == 0) or (self.global_epoch == self.nepochs-1)) or self.global_epoch == 5) and (self.global_epoch != 0):

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
        print("Start training generator")

        self.__training_stage__()

        print("Finish Trainig generator")

        self.writer.close()


    def __vis_comp_graph__ (self):
        """
        *******************************************************************
        __vis_comp_graph__ : visualising computational graph on tensorboard
        *******************************************************************
        """
        
        # iterate over data loader
        data = iter(self.vali_loader)
        # get the the first iteration
        (con_fl, seq_mels, _, _ ) = next(data)
        # computational graph visualization on tensorboard
        self.writer.add_graph(self.generator, (seq_mels.to(device),con_fl.to(device)))

        del data  # remove data iterator
        

    
    def __vis_lip_result__ (self):
        """
        ******************************************************
        __vis_lip_result__ : visualising result on tensorboard
        ******************************************************
        """

        data = iter(self.vali_loader)

        (con_fl, seq_mels , mel , gt_fl)  = next(data)
        

        with torch.no_grad() : 

            self.generator.eval()

            con_lip  = con_fl.to(device)
            gt_lip   = gt_fl.to(device)

            seq_mels = seq_mels.to(device)
            mel = mel.to(device)

            seq_len = con_lip.size(1)
    
            gen_lip, _ = self.generator(seq_mels, con_lip)
            
            # get one sample from a batch and convert to numpy array
            gen,ref, gt = gen_lip[0].detach().clone().cpu().numpy() ,con_lip[0].detach().clone().cpu().numpy(), gt_lip[0].detach().clone().cpu().numpy()
            # reshape to seq  
            gen = gen.reshape(seq_len,20,-1)
            ref = ref.reshape(seq_len,20,-1)
            gt  = gt.reshape(seq_len,20,-1)
            

            # plot single lip landmark
            single_fig = plot_lip_comparision(gen[0],ref[0],gt[0])
            # plot sequence of lip landmark
            seq_fig = []
            for idx in range(seq_len):
                
                vis_fig = plot_lip_comparision(gen[idx],ref[idx],gt[idx])

                seq_fig.append(vis_fig)
    
        
        return single_fig, seq_fig

    def __vis_vdo_result__ (self):
        """
        *********************************************************************
        __vis_vdo_result__ : visualising the result of the model on inference
        *********************************************************************
        """
        from .inference import Inference
        import argparse

        folder = "./results/training/{}/".format(self.save_name) 

        if not os.path.exists(folder):
            
            os.mkdir(folder)

        save_path = os.path.join(folder, "eval_epoch{}.mp4".format(self.global_epoch))
        print(save_path)

        parser = argparse.ArgumentParser(description="File for running Inference")

        parser.add_argument('--generator_checkpoint', type=str, help="File path for Generator model checkpoint weights" ,default='./checkpoints/generator/{}.pth'.format(self.save_name))

        parser.add_argument('--image2image_checkpoint', type=str, help="File path for Image2Image Translation model checkpoint weigths", default='./checkpoints/image2image/image2image.pth',required=False)

        parser.add_argument('--input_face', type=str, help="File path for input videos/images contain face",default='dummy/me.jpeg', required=False)

        parser.add_argument('--input_audio', type=str, help="File path for input audio/speech as .wav files", default='./dummy/main_testing.wav', required=False)

        parser.add_argument('--fps', type=float, help= "Can only be specified only if using static image default(25 FPS)", default=25,required=False)

        parser.add_argument('--fl_detector_batchsize',  type=int , help='Batch size for landmark detection', default = 32)

        parser.add_argument('--generator_batchsize', type=int, help="Batch size for Generator model", default=5)

        parser.add_argument('--output_name', type=str , help="Name and path of the output file", default=save_path)

        parser.add_argument('--vis_fl', type=bool, help="Visualize Facial Landmark ??", default=True)

        parser.add_argument('--test_img2img', type=bool, help="Testing image2image module with no lip generation" , default=False)

        args = parser.parse_args()

        eval_result = Inference(args=args)

        eval_result.start()




 
