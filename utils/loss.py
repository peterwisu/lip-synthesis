import torch
import torch.nn as nn
import numpy as np

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0, metric='euclidean'):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin

        self.metric = metric

    def forward(self,x1,x2,y):

        # calculate Euclidean distance 
        distance  = torch.nn.functional.pairwise_distance(x1,x2, p=2) # p=2 is norm degree for Euclidean distance

        term1 = (1-y) * torch.pow(distance,2)  # 

        term2 = (y) * torch.pow(torch.clamp(self.margin-distance, min=0),2)

        loss = torch.mean(term1 + term2)

        pred = np.array([ 1 if i > self.margin else 0 for i in distance ])
        label = y.detach().clone().cpu().numpy().reshape(-1).astype(int)
        acc = np.mean([1 if p == l else 0 for p,l in zip(pred,label) ]) * 100

        

        return loss , acc

class CosineBCELoss(torch.nn.Module):

    def __init__(self):
        super(CosineBCELoss, self).__init__()

        self.bce = torch.nn.BCELoss()

        
    def forward(self,x1,x2,y):
        
        # Get similarity score
        sim_score = torch.nn.functional.cosine_similarity(x1,x2)
        
        # compute BCE loss
        loss = self.bce(sim_score.unsqueeze(1),y)
        
        # get prediction 
        pred = np.array([1 if i >0.5 else 0 for i in sim_score])
        # clone gt from tensor
        label = y.detach().clone().cpu().numpy().reshape(-1).astype(int)
        # calculate accuracy
        acc = np.mean([1 if p == l else 0 for p,l in zip(pred,label) ]) * 100

        return loss, acc , pred, sim_score.detach().clone().cpu().numpy()

        

    

