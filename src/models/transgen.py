import torch
import torch.nn as nn
import numpy as np
import math




LIP_FEATURES = 40
AU_FEAT_SIZE = 128


class TransformerGenerator(nn.Module):

    def __init__ (self, hidden_size=256, in_size=80, dropout=0, lstm_size=AU_FEAT_SIZE, num_layers=3, num_window_frames=18, bidirectional=True ):
        super(TransformerGenerator, self).__init__()

        self.fc_in_features = 128#hidden_size * 2 if bidirectional else hidden_size

        
        self.au_encoder = nn.Sequential( #input (1,80,18)
                                        ConvBlock(1, 64, kernel_size=(3,3),stride=1, padding=0),
                                        ResidualBlock(64,64, kernel_size=(3,3),stride=1, padding=1),
                                        
                                        ConvBlock(64,128, kernel_size=(5,3), stride=(3,1), padding=1),
                                        ResidualBlock(128,128, kernel_size=(3,3), stride=(1,1), padding=1),

                                        ConvBlock(128,256, kernel_size=(5,3), stride=(3,3), padding=0),
                                        ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),

                                        ConvBlock(256,256, kernel_size=(3,3), stride=(3,3), padding=1),
                                        ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),
                                        
                                        ConvBlock(256,512, kernel_size=(3,2), stride=(1,1), padding=0),
                                    
                                        nn.Flatten()
                                        )


        encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                    nhead=8,
                                    dim_feedforward=2048,
                                    dropout=dropout,
                                    activation='relu'
                                    )

        self.trans = nn.TransformerEncoder(encoder_layer,num_layers=7)


        self.n_values = 60
        self.feed = nn.Sequential(
                LinearBlock(512+self.n_values, 512),
                LinearBlock(512, 256,),
                LinearBlock(256, 128,),  
                LinearBlock(128, 60 , dropout=False, batchnorm=False, activation=False),
                )


        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=5)

    def forward(self, au, lip, inference=False):
             
        # AU input shape : (B, Seq, 1, 80 , 18) 
     
        #  inputs shape ( B , seq, 20 , 3)
        lip = lip.reshape(lip.size(0),lip.size(1),-1) # outshape(Seq, B , 60) 
    
        # list for seq of extract features
        in_feats = []
        #  length of sequence
        seq_len = au.size(1)
        # batch_size
        batch_size = au.size(0)

        au =  au.reshape(batch_size * seq_len , 1 , 80 , -1) # (Batchsize * seq , 1 , 80 (num mel) , segments ) 

        in_feats = self.au_encoder(au)

        in_feats = in_feats.reshape(seq_len,batch_size,  -1)

        pos_out = self.pe(in_feats)

        trans_out = self.trans(pos_out)

        trans_out = trans_out.reshape(-1,trans_out.shape[-1])

        lip = lip.reshape(-1,lip.shape[-1])
        
        concat_input = torch.concat((trans_out,lip),dim=1)

        pred = self.feed(concat_input)

        pred = pred.reshape(batch_size, seq_len, self.n_values)

        return pred, lip



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LinearBlock(nn.Module):
    """
    Custom Linear Layer block with regularization (Dropout and Batchnorm) and Activation function 
    """
    def __init__ (self, in_features, out_features, dropout=True, dropout_rate=0.2, batchnorm=True, activation=True):
        
        super().__init__()

        self.mlp = nn.Linear(in_features = in_features,out_features = out_features) # Linear Layer 
        self.activation = nn.LeakyReLU(0.2) # activation function  layer 
        self.batchnorm = nn.BatchNorm1d(out_features) # Batch Normalization 1D layer 
        self.do_dropout = dropout # perform dropout 
        self.do_batchnorm = batchnorm # perform batchnorm
        self.do_activation = activation #  perform  activation 
        self.dropout   = nn.Dropout(dropout_rate) # Dropout rate 

    def forward(self, x):
        """
        forward propagation of this layer 
        """

        outs = self.mlp(x)


        if self.do_batchnorm:

            outs = self.batchnorm(outs)

        if self.do_activation:

            outs = self.activation(outs)

        if self.do_dropout:

            outs = self.dropout(outs)

        return outs



class ConvBlock(nn.Module):
    """
     Convolutional Layer (With batchnorm and activation)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):

        super().__init__()

        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                        nn.BatchNorm2d(out_channels),
                                        )

        self.activation = nn.ReLU()

    def forward(self, inputs):

        cnn_out = self.conv_layer(inputs)
        cnn_out = self.activation(cnn_out)

        return cnn_out

class ResidualBlock(nn.Module):
    """
        Convolutional Layers with Residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        
        super().__init__()

        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                        nn.BatchNorm2d(out_channels),
                                        )


        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                        nn.BatchNorm2d(out_channels),
                                        )

        self.activation = nn.ReLU()


    def forward(self,x):

        residual = x
        # first conv layer
        out = self.activation(self.conv_layer1(x))
        # second conv layer
        out = self.activation(self.conv_layer2(out))
        # residual connection
        out = out + residual

        return  out 

        















