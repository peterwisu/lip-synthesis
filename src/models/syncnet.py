import torch
import torch.nn as  nn
from utils.plot import plot_scatter_facial_landmark 
"""

SyncNet version for lip landmarks

"""
class SyncNet(nn.Module):
    def __init__(self,bilstm=True):
        super(SyncNet,self).__init__()
        
        self.bilstm = bilstm
        self.lip_hidden = 128
        self.n_values = 60 
        
        self.audio_encoder = nn.Sequential( #input (1,80,18)
                                        ConvBlock(1, 64, kernel_size=(3,3),stride=1, padding=0), # (78,16)
                                        ResidualBlock(64,64, kernel_size=(3,3),stride=1, padding=1,),
                                        ResidualBlock(64,64, kernel_size=(3,3),stride=1, padding=1,),
                                        
                                        ConvBlock(64,128, kernel_size=(5,3), stride=(3,1), padding=1,), # (26,16)
                                        ResidualBlock(128,128, kernel_size=(3,3), stride=(1,1), padding=1,),
                                        ResidualBlock(128,128, kernel_size=(3,3), stride=(1,1), padding=1,),

                                        ConvBlock(128,256, kernel_size=(5,3), stride=(3,3), padding=0,), # (8,5)
                                        ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1,),
                                        ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1,),

                                        ConvBlock(256,512, kernel_size=(3,3), stride=(3,3), padding=1,), # (3,2)
                                        ConvBlock(512,512, kernel_size=(3,2), stride=(1,1), padding=0,), # (3,2)
                                        nn.Flatten()
                                        )
        
       
        
        self.lip_encoder = nn.Sequential(
                                        LinearBlock(self.n_values, 512),
                                        nn.Dropout(0.2),
                                        LinearBlock(512,256),
                                        )
    

        self.visual_encoder =nn.LSTM(input_size=256,
                                  hidden_size=self.lip_hidden,
                                  num_layers=4, #2,
                                  batch_first=True,
                                  bidirectional=bilstm,
                                  )
        
        
        self.lip_size = 2 * self.lip_hidden if self.bilstm else self.lip_hidden  # output hidden size of lip lstm

        
        self.lip_fc = nn.Sequential(
            
            LinearBlock(self.lip_size, 1024),
            nn.Dropout(0.2),
            LinearBlock(1024, 512),
                )

    def forward(self, audio, lip):
        """
        forward propagation of this layer 
        """
        
        # lip shape (batch,seq,60) 
        # audio shape (batch,1,80,18)
        
        # extract features from melspectrogram
        au = self.audio_encoder(audio)
        
    
        
        lip_seq = lip.shape[1]
        batch_size= lip.shape[0]
        lip = lip.reshape(-1,self.n_values)
        
        # extract features from landmarks
        lip = self.lip_encoder(lip)
        lip = lip.reshape(batch_size,lip_seq, -1)
        
        # pass extracted lip features to BiLSTM
        vis_hidden, _ =self.visual_encoder(lip)
        
        # last hidden layers of lstm
        vis_hidden = vis_hidden[:,-1,:]
        
        # embeddings
        lip = self.lip_fc(vis_hidden)
        #au = self.audio_fc(au)
        
        # apply Euclidean(L2) norm
        au = nn.functional.normalize(au, p=2, dim=1)
        lip = nn.functional.normalize(lip, p=2, dim=1)

        return au, lip

  
class LinearBlock(nn.Module):
    """
    Custom Linear Layer block with regularization (Dropout and Batchnorm) and Activation function 
    """
    def __init__ (self, in_features, out_features,):
        
        super().__init__()
        
        self.linear_layer = nn.Sequential(
                nn.Linear(in_features,out_features),
                nn.BatchNorm1d(out_features),
                )

        self.activation = nn.ReLU()


    def forward(self, x):
        """
        forward propagation of this layer 
        """
        outs = self.activation(self.linear_layer(x))

        return outs



class ConvBlock(nn.Module):
    """
     Convolutional Layer (With batchnorm and activation)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=False, dropout_rate=0.2):

        super().__init__()

        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                        nn.BatchNorm2d(out_channels),
                                        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout2d(dropout_rate) if dropout else None


    def forward(self, inputs):
        """
        forward propagation of this layer 
        """
        cnn_out = self.conv_layer(inputs)
        cnn_out = self.activation(cnn_out)


        if self.dropout:

            cnn_out = self.dropout(cnn_out)

        return cnn_out

class ResidualBlock(nn.Module):
    """
        Convolutional Layers with Residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=False, dropout_rate=0.2):
        
        super().__init__()

        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                         nn.BatchNorm2d(out_channels),
                                        )


        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                         nn.BatchNorm2d(out_channels),
                                        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout else None


    def forward(self,x):
        """
        forward propagation of this layer 
        """
        residual = x
        # first conv layer
        out = self.activation(self.conv_layer1(x))
        # second conv layer
        out = self.activation(self.conv_layer2(out))
        # residual connection
        out = out + residual

        if self.dropout:

            out = self.dropout(out)

        return  out 

        











