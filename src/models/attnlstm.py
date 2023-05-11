import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LstmGen(nn.Module):

    def __init__ (self):
        super(LstmGen, self).__init__()

        self.n_values = 60 

        self.au_encoder = nn.Sequential( #input (1,80,18)
                                        ConvBlock(1, 64, kernel_size=(3,3),stride=1, padding=0),
                                        ResidualBlock(64,64, kernel_size=(3,3),stride=1, padding=1),
                                        
                                        ConvBlock(64,128, kernel_size=(5,3), stride=(3,1), padding=1),
                                        ResidualBlock(128,128, kernel_size=(3,3), stride=(1,1), padding=1),

                                        ConvBlock(128,256, kernel_size=(5,3), stride=(3,3), padding=0),
                                        ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),

                                        ConvBlock(256,256, kernel_size=(3,3), stride=(3,3), padding=1),
                                        ResidualBlock(256,256, kernel_size=(3,3), stride=(1,1), padding=1),
                                        
                                        ConvBlock(256,256, kernel_size=(3,2), stride=(1,1), padding=0),
                                    
                                        nn.Flatten()
                                        )


        self.lstm_encoder  = Encoder(input_size=256, hidden_size=256, num_layers=4, dropout=0.25, bidirectional=True, batch_first=False)

        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=5)

        self.self_attn = nn.MultiheadAttention(512, num_heads=8)

        self.feed = nn.Sequential(
                LinearBlock(512+self.n_values, 256),
                LinearBlock(256, 128),
                LinearBlock(128, 60 , dropout=False, batchnorm=False, activation=False),
                )


    def forward(self, au, lip):
        


        # AU input shape : (B, Seq, 1, 80 , 16) 
     
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

        lstm_outs , hidden, cell = self.lstm_encoder(in_feats) 

        pos_out = self.pe(lstm_outs)


        attn_out, attn_weight = self.self_attn(pos_out,pos_out,pos_out)#[0]
        


        attn_out = attn_out.reshape(-1,attn_out.shape[-1])

        lip = lip.reshape(-1,lip.shape[-1])
        
        concat_input = torch.concat((attn_out,lip),dim=1)

        pred = self.feed(concat_input)

        pred = pred.reshape(batch_size, seq_len, self.n_values)


        return pred , lip 


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


class PositionalEncoding(nn.Module):
    """"
    Positional Encoding from Pytorch website 
    """

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

        








