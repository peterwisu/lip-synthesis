import torch
from torch import nn
from torch.nn import functional as F
"""

SyncNet version for Lip key point

"""
class ModSyncNet(nn.Module):
    def __init__(self, inp='2d',bilstm=True):
        super(ModSyncNet,self).__init__()
        
        self.bilstm = bilstm
        self.au_hidden =  256
        self.lip_hidden = 128
        
         
        self.audio_encoder = nn.LSTM(input_size=80,
                                     hidden_size=self.au_hidden,
                                     num_layers=2,
                                     batch_first=True,
                                     bidirectional=self.bilstm
                                    )

        self.lip_encoder =nn.LSTM(input_size=40,
                                  hidden_size=self.lip_hidden,
                                  num_layers=2,
                                  batch_first=True,
                                  bidirectional=bilstm
                                  )
        
        
        
        self.au_size = 2* self.au_hidden if self.bilstm else self.au_hidden
        
        self.lip_size = 2 * self.lip_hidden if self.bilstm else self.lip_hidden 
        self.lip_fc = nn.Sequential(
            
            Linear(self.lip_size, 1024),
            Linear(1024, 512),
            Linear(512, 256)
                )
        
        self.audio_fc = nn.Sequential(
                        Linear(self.au_size,1024),
                        Linear(1024,512),
                        Linear(512,256)
        )          
    

    def forward(self, audio, lip):
        
        audio =  audio.reshape(-1,18,80)
        #lip   =  lip.reshape(-1,5,40)
      
        
        au_emb, _ = self.audio_encoder(audio)
        lip_emb, _ =self.lip_encoder(lip)


        au_emb = au_emb[:,-1,:]
        lip_emb = lip_emb[:,-1,:]

    

        au = self.audio_fc(au_emb)
        lip = self.lip_fc(lip_emb)

        au = nn.functional.normalize(au, p=2, dim=1)
        lip = nn.functional.normalize(lip, p=2, dim=1)
        
        
        return au, lip

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
    

class Linear(nn.Module):
    def __init__(self, cin, cout, dropout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear = nn.Sequential(
                        nn.Linear(cin, cout),
                        nn.BatchNorm1d(cout)
                        )

        self.activation = nn.ReLU()

        self.dropout = dropout

    

        self.dropout_layer = nn.Dropout(0.20) #0.5

    def forward(self, x):

        outputs = self.linear(x)


        if self.dropout:

            return self.dropout_layer(self.activation(outputs))

        else:

            return self.activation(outputs)

