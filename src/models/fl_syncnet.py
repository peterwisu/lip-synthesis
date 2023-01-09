import torch
from torch import nn
from torch.nn import functional as F


"""

SyncNet version for Lip key point

"""


class SyncNet_fl(nn.Module):

    def __init__(self):
        super(SyncNet_fl, self).__init__()

        self.audio_encoder = nn.Sequential(

            Conv2d(1, 96, kernel_size=3, stride=1, padding=1),  # 80 x 18
            # Conv2d(96,96, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), # 40 x 9
            Conv2d(96, 256, kernel_size=3, stride=(3, 1), padding=1),  # 14 x 9
            # Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.MaxPool2d(2, 2), # 7 x 4
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),   # 3 x 2
            Conv2d(512, 2048, kernel_size=3, stride=(3, 1), padding=1,),  # 2x2
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.lip_fc = nn.Sequential(
            
            Linear(300, 1024, dropout=True),
            Linear(1024, 2048, dropout=True),
            Linear(2048, 4096, dropout=True),
            Linear(4096, 256)
            )

        self.audio_fc = nn.Sequential(

            Linear(2048, 4096, dropout=True),
            Linear(4096, 256)
        )


    def forward(self,audio,lip_fl):

        # Reshape landmark into 1 dimension
        lip_fl = lip_fl.reshape(lip_fl.shape[0], -1)
        
        # encode embedding 
        lip_embedding = self.lip_fc(lip_fl)
        audio_embedding = self.audio_encoder(audio)
        audio_embedding = self.audio_fc(audio_embedding)

        lip_embedding = lip_embedding.view(lip_embedding.size(0), -1)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        
        lip_embedding = F.normalize(lip_embedding, p=2, dim=1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        return audio_embedding, lip_embedding


class Linear(nn.Module):
    def __init__(self, cin, cout, dropout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear = nn.Sequential(
                        nn.Linear(cin, cout),
                        nn.BatchNorm1d(cout)
                        )

        self.activation = nn.ReLU()

        self.dropout = dropout

        self.dropout_layer = nn.Dropout(0.50)

    def forward(self, x):

        outputs = self.linear(x)

        if self.dropout:

            return self.dropout_layer(self.activation(outputs))

        else:

            return self.activation(outputs)


# Convolutional Layer
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
