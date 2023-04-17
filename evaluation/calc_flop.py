import sys

sys.path.append('../')
from pthflops import count_ops
import torch
from src.models.attnlstm import LstmGen as Attnlstm
from src.models.lstmgen import LstmGen as Lstm
from src.models.syncnet import SyncNet
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info



def prepare_input_gen(resolution):
    au = torch.FloatTensor(1,5,1,80,18)
    lip = torch.FloatTensor(1,5,20,3)
  
    return dict({'au' :au , 'lip': lip})


def prepare_input_syncnet(resolution):
  
    audio = torch.FloatTensor(1,1,80,18)
    lip = torch.FloatTensor(1,5,60)
  
    return dict({'audio' :audio , 'lip': lip})

with torch.cuda.device(0):
    attnlstm_model = Attnlstm()

    lstm_model = Lstm()
    
    syncnet = SyncNet()
    
    attn_lstm_macs, attn_lstm_params = get_model_complexity_info(attnlstm_model, ((1,5,1,80,18),(1,5,20,3)),
                                            input_constructor=prepare_input_gen, 
                                            as_strings=True, print_per_layer_stat=True, verbose=True, )
    
    lstm_macs, lstm_params = get_model_complexity_info(lstm_model, ((1,5,1,80,18),(1,5,20,3)),
                                            input_constructor=prepare_input_gen, 
                                            as_strings=True, print_per_layer_stat=True, verbose=True, )
    
    
    syncnet_macs, syncnet_params = get_model_complexity_info(syncnet, ((1,5,1,80,18),(1,5,20,3)),
                                            input_constructor=prepare_input_syncnet, 
                                            as_strings=True, print_per_layer_stat=True, verbose=True, )
    

    
    print('{:<30}  {:<8}'.format('Syncnet Computational complexity: ', syncnet_macs))
    print('{:<30}  {:<8}'.format('Syncnet Number of parameters: ', syncnet_params))
    print()
    print('{:<30}  {:<8}'.format('Attn_lstm Computational complexity: ', lstm_macs))
    print('{:<30}  {:<8}'.format('Attn_lstm Number of parameters: ', lstm_params))
    print()
    print('{:<30}  {:<8}'.format('Attn_lstm Computational complexity: ', attn_lstm_macs))
    print('{:<30}  {:<8}'.format('Attn_lstm Number of parameters: ', attn_lstm_params))







