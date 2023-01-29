import torch
import torch.nn as  nn



class Linear(nn.Module):
    """
    Custom Linear Layer block with regularization (Dropout and Batchnorm) and Activation function 
    """
    def __init__ (self, in_features, out_features, dropout=True, dropout_rate=0.2, batchnorm=True, activation=True):
        """


        """
        
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

        if self.do_activation:

            outs = self.activation(outs)

        if self.do_batchnorm:

            outs = self.batchnorm(outs)

        if self.do_dropout:

            outs = self.dropout(outs)

        return outs



class Conv_block(nn.Module):
    """
     Convolutional Layer (With batchnorm and dropout)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, do_dropout=False,
                 do_batchnorm=False, dropout_rate=0.25, do_residual=False):

        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.activation = nn.ReLU()
        self.do_dropout = do_dropout
        self.do_batchnorm = do_batchnorm
        self.do_residual = do_residual
        if self.do_dropout:

            self.drop_layer = nn.Dropout2d(dropout_rate)

        if self.do_batchnorm:
            self.batchnorm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):

        cnn_out = self.conv_layer(inputs)
        cnn_out = self.activation(cnn_out)

        if self.do_batchnorm:
            cnn_out = self.batchnorm_layer(cnn_out)

        if self.do_residual:

            cnn_out +=  inputs

        if self.do_dropout:
            cnn_out = self.drop_layer(cnn_out)

        return cnn_out





