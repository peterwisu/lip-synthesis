"""

"""

import torch
import torch.nn as nn
import time 


class LstmGen(nn.Module):

    def __init__ (self, is3D=True):
        super(LstmGen, self).__init__()


        self.n_values = 60 if is3D else 40 # values of landmarks 60 if it is 3D ( 20 x 3) else 2D is 40 (20 x 2)

        self.feed = nn.Sequential(
                LinearBlock(1024 + self.n_values, 512),
                LinearBlock(512, 256,),
                LinearBlock(256, 128,),  
                LinearBlock(128, 60 , dropout=False, batchnorm=False, activation=False),
                #nn.Linear(128,self.n_values)
                )
        
        self.au_encoder = nn.Sequential( #input (1,80,18)
                                        ConvBlock(1, 64, kernel_size=(3,3),stride=1, padding=0),
                                        #ResidualBlock(64,64, kernel_size=(3,3),stride=1, padding=1),
                                        #ResidualBlock(64,64, kernel_size=(3,3),stride=1, padding=1, ),
                                        
                                        ConvBlock(64,128, kernel_size=(5,3), stride=(3,1), padding=1),
                                        #ResidualBlock(128,128, kernel_size=(3,3), stride=(1,1), padding=1),
                                        #ResidualBlock(128,128, kernel_size=(3,3), stride=(1,1), padding=1),

                                        ConvBlock(128,256, kernel_size=(5,3), stride=(3,3), padding=0),
                                        #ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),
                                        #ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),

                                        ConvBlock(256,256, kernel_size=(3,3), stride=(3,3), padding=1),
                                        #ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),
                                        #ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),
                                        
                                        ConvBlock(256,512, kernel_size=(3,2), stride=(1,1), padding=0),
                                    
                                        nn.Flatten()
                                        )



        self.lstm_encoder  = Encoder(input_size=512, hidden_size=512, num_layers=4, dropout=0.25, bidirectional=True, batch_first=True)


    def forward(self, au, lip, inference=False):

    
        if inference:

            outs = []
            print(au.shape)

            seq_len = au.size(0)  # for inferece batch_size and seq is the same
            # inshape (B or Seq , 80, 18)
            au =   au.unsqueeze(1) # outshape (B or Seq, 1, 80, 18)

            out =  self.au_encoder(au) # (B or Seq, 256)
 
            out = out.unsqueeze(0)
            
            lstm_out, hidden , cell = self.lstm_encoder(out)

            """ 
            # Out Shape of lstm encoder
            # 
            # en : (seq,  B , hidden_size *2 )) 
            # hidden : (num_lay * 2, B, hidden_size)
            # cell : (num_lay * 2 , B , hidden_size)
            """

            lstm_out = lstm_out.squeeze(0)

            # concat feature with lip
            concat_input =torch.cat((lstm_out,lip),dim=1)
            
            outs = self.feed(concat_input)

        
        else:
            
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

            in_feats = in_feats.reshape(batch_size, seq_len, -1)


            lstm_out, hidden, cell = self.lstm_encoder(in_feats) 

            """ 
            # Out Shape of lstm encoder
            # 
            # en : (B, seq, , hidden_size *2 )) 
            # hidden : (num_lay * 2, B, hidden_size)
            # cell : (num_lay * 2 , B , hidden_size)
            """


            # concat feature with lip
            concat_input =torch.cat((lstm_out,lip),dim=2)

            concat_input = concat_input.reshape(-1,concat_input.size(2)) # (B * seq , 1064)
            
            outs = self.feed(concat_input)

            outs = outs.reshape(batch_size ,seq_len, -1)



        return outs , lip 


class Encoder(nn.Module):

    def __init__ (self, input_size, hidden_size, num_layers, dropout,bidirectional=True,batch_first=False):

        super(Encoder,self).__init__()

        self.lstm = nn.LSTM(
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                bidirectional=bidirectional,
                                batch_first=batch_first
                            )

    def forward(self, inputs):

        out,  (hidden, cell) = self.lstm(inputs)

        return out , hidden , cell 


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

        










