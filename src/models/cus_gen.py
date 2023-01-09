import torch
import torch.nn as nn
import numpy as np
from torch.nn import MaxPool2d




LIP_FEATURES = 40
AU_FEAT_SIZE = 128


class Generator(nn.Module):

    def __init__ (self, hidden_size=256, in_size=80, dropout=0, lstm_size=AU_FEAT_SIZE, num_layers=3, num_window_frames=18, bidirectional=True ):
        super(Generator, self).__init__()

        self.fc_in_features = 128#hidden_size * 2 if bidirectional else hidden_size


        self.au_encoder = nn.Sequential( #Input ( 1, 80 ,18)
                    nn.Conv2d(1,16, kernel_size = (3,3), stride=1, padding=0), # (78,16)
                    nn.ReLU(),
                    nn.Conv2d(16,32, kernel_size = (5,3), stride=(3,1), padding=1), # (26,16)'              
                    nn.ReLU(),
                    nn.Conv2d(32,64, kernel_size = (5,3), stride=(3,3), padding=0), # (8,5)
                    nn.ReLU(),
                    nn.Conv2d(64,128, kernel_size = (3,3), stride=(3,3), padding=1), #(3,2)
                    nn.ReLU(),
                    nn.Conv2d(128,256, kernel_size = (3,2), stride=1, padding=0), #(1,1)
                    nn.Relu(),
                    nn.Flatten()

                                        )



        encoder_layer = nn.TransformerEncoderLayer(d_model=256,
                                    nhead=8,
                                    dim_feedforward=2048,
                                    dropout=dropout,
                                    activation='relu'
                                    )

        self.trans = nn.TransformerEncoder(encoder_layer,num_layers=7)



        # input (b, 256 +60) 
        self.fc_layers = nn.Sequential(
                                Linear((256 + LIP_FEATURES), 512),
                                Linear(512, 256,),  # dropout=True),
                                Linear(256, 128,),  # dropout=True),
                                # Linear(128, 64,),
                                Linear(128, 60 , dropout=False, batchnorm=False, activation=False)
        )  #  dropout=True))

    def forward(self, au, lip, inference=False):
             
        if inference: 
 
            # inshape (B or Seq , 80, 18)
            au =   au.unsqueeze(1) # outshape (B or Seq, 1, 80, 18)

            out =  self.au_encoder(au) # (B or Seq, 256)

            out =  torch.cat((out , lip), dim=1) # (B or Seq , Hidden + lip)

            all_lip =  self.fc_layers(out)  # (B or Seq , 60)

        else:  

            
            #  inputs shape ( B , seq, 20 , 3)
            lip = lip.reshape(lip.size(1),lip.size(0),-1) # outshape(Seq, B , 60) 
        
            # list for seq of extract features
            au_feats = []

            # au shape (B,seq,1,80,18)
            for idx in range(au.size(1)): # iterate through seq

                # innput shape (B,1,80,18) 
                out = self.au_encoder(au[:,idx]) #outshape (B,256)
        
                au_feats.append(out)
            
            au_feats = torch.stack(au_feats, dim=0) # outshape ( Seq, B ,256)            
            
            # input shape (Seq, B, 256)
            trans_out = self.trans(au_feats) # outshape (Seq, B, 256)

            # list of generated lip            
            all_lip = []

            for  idx in range(trans_out.size(0)): # Loop through seq
                
                # concat the embedding with lip landmarks
                inputs = torch.cat((trans_out[idx],lip[idx]),dim=1)
                
                # input shape (B, Hidden + FL)
                out = self.fc_layers(inputs) # outshape (B, 60)
                
                all_lip.append(out)
                 
            all_lip = torch.stack(all_lip,dim=0) # Outshape (Seq, B , 60) 

            all_lip = all_lip.reshape(all_lip.size(1),all_lip.size(0),-1)  # Outshape (B, Seq, 60)

         
        return all_lip, lip

"""
 Linear Layers
"""
class Linear(nn.Module):
    def __init__ (self, in_features, out_features, dropout=False, dropout_rate=0.2, batchnorm=True, activation=True):
        super().__init__()

        self.mlp = nn.Linear(in_features = in_features,out_features = out_features)

        self.activation = nn.LeakyReLU(0.2)
        #self.activation = nn.LeaReLU()
        self.batchnorm = nn.BatchNorm1d(out_features)

        self.do_dropout = dropout
        self.do_batchnorm = batchnorm
        self.do_activation = activation
        self.dropout   = nn.Dropout(dropout_rate)

    def forward(self, x):

        outs = self.mlp(x)
        if self.do_activation:

            outs = self.activation(outs)

        if self.do_batchnorm:

            outs = self.batchnorm(outs)

        if self.do_dropout:

            outs = self.dropout(outs)

        return outs






















