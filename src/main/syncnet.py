
#from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
import numpy as np
import os
from hparams import hparams
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from src.models.syncnet import SyncNet
from src.dataset.syncnet import Dataset
from utils.wav2lip import save_checkpoint, load_checkpoint
from utils.utils import save_logs, load_logs
from utils.plot import plot_comp, plot_roc, plot_cm , plot_single
from utils.loss import CosineBCELoss


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class TrainSyncNet():
    """
    *****************************************************************************
    TrainSyncNet : Training pretrain SyncNet model (Expert LipSync Discriminator)
    *****************************************************************************
    @author : Wish Suharitdarmong
    """
    
    def __init__(self,args):
        
        # arguments and hyperparameters 
        self.save_name  = args.save_name
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_path = args.checkpoint_path
        self.batch_size = hparams.syncnet_batch_size
        self.global_epoch = 0
        self.nepochs = hparams.syncnet_nepochs
        self.do_train  = args.do_train
        
        
        
        # if create checkpoint dir if it does not exist
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
        
        # Tensorboard
        self.writer = SummaryWriter("../tensorboard/{}".format(self.save_name))
         
        # Dataset 
        
        
        # if not training stage then do not load training and validations set 
        if self.do_train:
            self.train_dataset = Dataset(split='train', args=args)

            self.vali_dataset  = Dataset(split='val', args=args)
            
            self.train_loader = data_utils.DataLoader(self.train_dataset,
        
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=hparams.num_workers)

            self.vali_loader = data_utils.DataLoader(self.vali_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=hparams.num_workers)
            
        # Load Testing Set 
        self.test_dataset = Dataset(split='test',args=args)
    
        
        self.test_loader = data_utils.DataLoader(self.test_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=hparams.num_workers)
         
         
        # SyncNet Model 
        self.model = SyncNet().to(device)

        print(self.model)
        # optimizer 
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr,
                           )
        
        # Loss/Cost/Objective function 
        self.bce_loss = nn.BCELoss()
        
        
        # load checkpoint if the path is given 
        self.continue_ckpt = False
        if self.checkpoint_path is not None:
            self.continue_ckpt = True
            self.model, self.optimizer, self.global_epoch = load_checkpoint(self.checkpoint_path, self.model, self.optimizer,use_cuda, reset_optimizer=False)
            
            print("Load Model from Checkpoint Path")
            
            
        # If not training dont load training logs 
        if  self.do_train :
            
            # if contutinuing from checkpoint then log previous log if not create new empty logs
            if self.continue_ckpt:

                print("Starting Checkpoint from epochs : ", self.global_epoch)

                self.train_loss , self.train_acc,  self.vali_loss, self.vali_acc = load_logs(model_name = "syncnet", savename="{}.csv".format(self.save_name), epoch=self.global_epoch, type_model="syncnet")
                self.global_epoch +=1
            else:

                print("Not Continue from Checkpoint")
                self.train_loss = np.array([])
                self.vali_loss = np.array([])
                self.train_acc = np.array([])
                self.vali_acc = np.array([])
            
        # If GPU detect more that one then train model in parallel 
        if torch.cuda.device_count() > 1:
             
            self.model = DataParallel(self.model)
            self.batch_size = self.batch_size * torch.cuda.device_count()
            print("Training or Testing model with  {} GPU " .format(torch.cuda.device_count()))
            
        self.model.to(device)     
        
        

        self.cosine_bce = CosineBCELoss()
    


    def __train_model__(self):
        """
        ********************************************
        __train_model__ : Training stage for SyncNet  
        ********************************************
        @author : Wish Suharitdarmong
        """
 
        running_loss = 0. 
        running_acc = 0.
        iter_inbatch = 0
        prog_bar = tqdm(self.train_loader)

        for (x, mel, y) in prog_bar:

            # X shape : (B, 5, 20, 3) -- 5 consecutive lip landmarks
            # Mel shape : (B, 1 , 80, 18) -- Melspectrogram features 

            self.model.train()
            self.optimizer.zero_grad()
            # allocate data to CUDA

            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)
            
            a , v = self.model(mel, x)

            loss, acc,_ ,_ = self.cosine_bce(a,v,y)

            loss.backward() # Backprop
            self.optimizer.step() # Gradient descent step

            running_loss += loss.item()
            running_acc  += acc
            iter_inbatch +=1
            

            prog_bar.set_description('TRAIN EPOCHS : {} Loss: {:.3f} Accuracy: {:.3f}'.format(self.global_epoch,
                                                                                running_loss / iter_inbatch,
                                                                                running_acc / iter_inbatch))

        avg_loss = running_loss / iter_inbatch
        avg_acc = running_acc / iter_inbatch


        return avg_loss, avg_acc

    
    def __eval_model__ (self,split=None):
        """
        
        """

        running_loss = 0.
        running_acc = 0.
        iter_inbatch = 0

        # check which data spilt will be use for validation
        if split == 'test':

            prog_bar = tqdm(self.test_loader)

        elif split == 'val':

            prog_bar = tqdm(self.vali_loader)

        else :
            # throw error if spilt does not exist
            raise ValueError("Wrong data spilt for __eval_model__ , only vali and test are accept in model evaluation")

        # array for visualizing CM and ROC
        y_pred_label = np.array([])
        y_pred_proba = np.array([])
        y_gt =  np.array([])


        with torch.no_grad():

            for (x, mel, y) in prog_bar:

                self.model.eval()

                x = x.to(device)
                mel = mel.to(device)
                y = y.to(device)

                a, v =  self.model(mel, x)


                loss, acc,pred_label,pred_proba = self.cosine_bce(a,v,y)

                # add label and proba to a array
                y_pred_label = np.append(y_pred_label, pred_label)
                y_pred_proba = np.append(y_pred_proba, pred_proba)
                y_gt   = np.append(y_gt, y.clone().detach().cpu().numpy())



                running_loss += loss.item()
                running_acc += acc
                iter_inbatch +=1
                

                prog_bar.set_description('EVAL EPOCHS : {} Loss: {:.3f} Accuracy: {:.3f}'.format(self.global_epoch,
                                                                                          running_loss / iter_inbatch,
                                                                                          running_acc / iter_inbatch
                                                                                         ))

        # plot roc and cm
        roc_fig = plot_roc(y_pred_proba,y_gt)
        cm_fig =  plot_cm(y_pred_label,y_gt)

        avg_loss = running_loss / iter_inbatch
        avg_acc = running_acc / iter_inbatch

        return avg_loss, avg_acc , cm_fig, roc_fig
    
    
    def __update_logs__ (self, cur_train_loss, cur_train_acc, cur_vali_loss, cur_vali_acc, cm_fig, roc_fig):
        """
        *************************************
        __update_logs__ : update training log 
        *************************************
        @author : Wish Suharitdarmong
        """
        
        # Logs in array
        self.train_loss = np.append(self.train_loss, cur_train_loss)
        self.train_acc  = np.append(self.train_acc , cur_train_acc)
        self.vali_loss  = np.append(self.vali_loss, cur_vali_loss)
        self.vali_acc   = np.append(self.vali_acc, cur_vali_acc)

        # save logs in csv file
        save_logs(train_loss = self.train_loss,
                      train_acc  = self.train_acc,
                      vali_loss  = self.vali_loss,
                      vali_acc   = self.vali_acc, 
                      model_name="syncnet",savename='{}.csv'.format(self.save_name))

        # ******* plot metrics *********
        # plot metrics comparison (train vs validation)
        loss_comp  = plot_comp(self.train_loss,self.vali_loss,  name="Loss")
        acc_comp = plot_comp(self.train_acc, self.vali_acc, name="Accuracy")
        # plot individually
        train_loss_plot = plot_single(self.train_loss, 'train_loss',name="Train Loss")
        vali_loss_plot  = plot_single(self.vali_loss, 'vali_loss', name='Validation Loss')
        train_acc_plot  = plot_single(self.train_acc, 'train_acc', name="Train Accuracy")
        vali_acc_plot   = plot_single(self.vali_acc,  'vali_acc' , name='Validation Accuracy')
            
        # ******** Tensorboard ********** 
        # Scalar 
        self.writer.add_scalar("Optim/Lr", self.optimizer.param_groups[0]['lr'],self.global_epoch)
        self.writer.add_scalar('Loss/train', cur_train_loss , self.global_epoch)
        self.writer.add_scalar('Loss/vali',  cur_vali_loss,   self.global_epoch)
        self.writer.add_scalar('Acc/train',  cur_train_acc,   self.global_epoch)
        self.writer.add_scalar('Acc/vali',   cur_vali_acc,    self.global_epoch) 
        # Figure
        self.writer.add_figure('Comp/loss', loss_comp, self.global_epoch)
        self.writer.add_figure('Comp/acc', acc_comp, self.global_epoch)
        self.writer.add_figure('Train/acc', train_acc_plot, self.global_epoch)
        self.writer.add_figure('Train/loss', train_loss_plot, self.global_epoch)
        self.writer.add_figure('Vali/acc' , vali_acc_plot, self.global_epoch)
        self.writer.add_figure('Vali/loss', vali_loss_plot, self.global_epoch)
        self.writer.add_figure("Vis/confusion_matrix", cm_fig, self.global_epoch)
        self.writer.add_figure("Vis/ROC_curve", roc_fig, self.global_epoch)
        


    def __training_stage__ (self):
        """
        
        """

        while self.global_epoch < self.nepochs:
            
            # train model
            cur_train_loss , cur_train_acc  = self.__train_model__()
            # validate model
            cur_vali_loss , cur_vali_acc , cm_fig, roc_fig = self.__eval_model__(split='val')
            
            self.__update_logs__(cur_train_loss, cur_train_acc, cur_vali_loss, cur_vali_acc, cm_fig, roc_fig)

            # Save model checkpoint
            save_checkpoint(self.model, self.optimizer, self.checkpoint_dir, self.global_epoch, self.save_name)
            # increment global epoch     
            self.global_epoch +=1


    def start_training(self):
        """
        
        Training Model 
        
        """
        print("Save name : {}".format(self.save_name))
        print("Using CUDA : {} ".format(use_cuda))
        
        if use_cuda: print ("Using {} GPU".format(torch.cuda.device_count())) 
        
        print("Training dataset {}".format(len(self.train_dataset)))
        print("Validation dataset {}".format(len(self.vali_dataset)))
        print("Testing dataset {}".format(len(self.test_dataset)))

        print("Start training SyncNet")
        
        self.__training_stage__()

        print("Finish Trainig SyncNet")

        print(" Evaluating SyncNet with Test Set")
        
        # evaluate model on test set
        test_loss, test_acc, cm_fig, roc_fig = self.__eval_model__(split='test')

        print("Testing Stage")
        print("Loss : {} , Acc : {} ".format(test_loss, test_acc))
        # plot metrics for test set on tensorboard
        self.writer.add_figure("Test/confusion_matrix",cm_fig,0)
        self.writer.add_figure("Test/ROC_curve",roc_fig,0)
        self.writer.add_scalar("Test/loss",test_loss,0)
        self.writer.add_scalar("Test/acc",test_acc,0)

        self.writer.close()


    def start_testing(self):
        """
        
        Testing model 
        
        """ 
        
        print("Using CUDA : {} ".format(use_cuda))

        if use_cuda: print ("Using {} GPU".format(torch.cuda.device_count())) 
        
        print("Testing dataset {}".format(len(self.test_dataset)))
        
        test_loss, test_acc, cm_fig, roc_fig = self.__eval_model__(split='test')

        print("Testing Stage")
        print("Loss : {} , Acc : {} ".format(test_loss, test_acc))

        

        

        

        

        
            
        


            








            














            






        
        
        
    
        
        
        
        
        
        
        
        
