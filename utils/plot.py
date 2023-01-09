import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import cv2


def plot_single(val, label, name):
    
    fig , (ax)  = plt.subplots(nrows=1,ncols=1)
    
    ax.plot(val, color='r', label=label, linestyle='-')
    if "loss" in name.lower():
        ax.legend(loc="upper right")
    if "accuracy" in name.lower():
        ax.legend(loc="lower right")
    ax.set_title("{}".format(name))
    figure = ax.get_figure()
    plt.close(fig)
    
    return figure
    

def plot_comp(train,vali, name):
    """
    ************************************************************
    plot_comp : Plot comparision between training and evaluation
    ************************************************************
    @author: Wish Suharitdamrong
    ------
    inputs
    ------
        train :  array containing a training logs value (accuracy or loss)
        vali :  array containg a evaluation log value 
    -------
    returns
    -------
        figure : Plot comparision figure
    """
    
 
    fig, (ax) = plt.subplots(nrows=1,ncols=1)
    ax.plot(train, color='r' , label='Train', linestyle='-')
    ax.plot(vali, color='b', label='Validation', linestyle='-')
    if "loss" in name.lower():
        ax.legend(loc="upper right")
    if "accuracy" in name.lower():
        ax.legend(loc="lower right")
    ax.set_title("{}".format(name))
    figure = ax.get_figure()
    plt.close(fig)
    
    return figure






def plot_cm(y_pred_label,y_gt):
    """
    ****************************
    Plot Confusion Matrix Figure
    ****************************
    @author: Wish Suharitdamrong
    ----------
    parameters
    ----------    
        y_pred_label :  labels predicted from a model
        y_gt  : grouth truth of each labels
    -------
    returns
    -------
        dia_cm : Figure of Confusion Matrix
    """

    df_cols_rows = ["Positive", "Negative"]

    cm = confusion_matrix(y_gt, y_pred_label,normalize="all")
    cm_df = pd.DataFrame(cm , index = df_cols_rows , columns = df_cols_rows)
    
    # Plot Fig 
    fig, (ax) = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Confusion Maxtrix")
    dia_cm = sns.heatmap(cm_df, annot =True).get_figure() 
    plt.close(fig)
        
    return dia_cm




def plot_roc(y_pred_proba,y_gt):
    """
    **************************************************
    Receiver Operating Characteristic (ROC) Curve plot
    **************************************************
    @authors: Wish Suharitdamrong
    ---------
    parameter
    ---------
        y_pred_proba : Probabilities of each labels predicted from a model
        y_gt : Ground truth of eachs labels
    -------
    returns
    -------
        dia_roc : ROC Curve Figure
    """
    
    # Calculate False and True positive rate
    fpr, tpr, threshold = roc_curve(y_gt, y_pred_proba)
    
    # Calculate area under a curve 
    roc_auc = auc(fpr, tpr)
    
    # Plot Fig 
    fig, (ax) = plt.subplots(nrows=1, ncols=1)
    # Plot curve
    line1 = ax.plot(fpr, tpr, 'b', label= "AUC = {:.2f}".format(roc_auc))
    # Plot Threshold
    line2 = ax.plot([0,1],[0,1],'--r', label = "Random Classifier")
    ax.legend(loc="lower right")
    ax.set_title("Receiving Operating Characteristic (ROC) Curve")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate") 
    dia_roc = ax.get_figure()
    plt.close(fig) 
     
    return dia_roc



def plot_compareLip(predict,gt_lip):

    lip_pred = predict.reshape(predict.shape[0], predict.shape[1], 20, -1)[0, 0].detach().clone().cpu().numpy()
    fl = gt_lip.reshape(predict.shape[0], predict.shape[1], 20, -1)[0, 0].detach().clone().cpu().numpy()

    fig, (ax) = plt.subplots(nrows=1, ncols=1)

    # Plot Generated Lip
    ax.scatter(lip_pred[:, 0], lip_pred[:, 1], s=20, c='r', linewidths=2)
    ax.plot(lip_pred[0:7, 0], lip_pred[0:7, 1], c="tab:red", linewidth=3 , label='Generated')
    ax.plot(np.append(lip_pred[6:12, 0], lip_pred[0, 0]), np.append(lip_pred[6:12, 1], lip_pred[0, 1]), c="tab:red", linewidth=3)
    ax.plot(lip_pred[12:17, 0], lip_pred[12:17, 1], c="tab:red", linewidth=3)
    ax.plot(np.append(lip_pred[16:20, 0], lip_pred[12, 0]), np.append(lip_pred[16:20, 1], lip_pred[12, 1]), c="tab:red", linewidth=3)

    # Plot Ground Truth Lip
    ax.scatter(fl[:, 0], fl[:, 1], s=20, c='g', linewidths=2)
    ax.plot(fl[0:7, 0], fl[0:7, 1], c="tab:blue", linewidth=3, label='Ground Truth')
    ax.plot(np.append(fl[6:12, 0], fl[0, 0]), np.append(fl[6:12, 1], fl[0, 1]), c="tab:blue", linewidth=3)
    ax.plot(fl[12:17, 0], fl[12:17, 1], c="tab:blue", linewidth=3)
    ax.plot(np.append(fl[16:20, 0], fl[12, 0]), np.append(fl[16:20, 1], fl[12, 1]), c="tab:blue", linewidth=3)
    ax.legend(loc="upper left")
    ax.invert_yaxis()
    com_fig = ax.get_figure()
    plt.close(fig)

    return com_fig



def plot_visLip(fl):


    fl[:,1] = -fl[:,1]

    fig, ax = plt.subplots()
   
    ax.scatter(fl[:,0],fl[:,1],s=20, c='r',linewidths=4)
    ax.plot(fl[0:7,0],fl[0:7,1], c="tab:pink", linewidth=3 )
    ax.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:pink", linewidth=3 )
    ax.plot(fl[12:17,0],fl[12:17,1], c="tab:pink", linewidth=3 )
    ax.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:pink", linewidth=3)
    ax.axvline(x = 1, color = 'b', linestyle='--',label = 'axvline - full height')

    vis_fig = ax.get_figure()
    plt.close(fig)
    
    return vis_fig


def plot_seqlip_comp(pred,ref,gt):

    
    fig, (ax1, ax2) = plt.subplots(nrows=1 , ncols=2, figsize=(15,6))

    fl = pred
    ax1.scatter(fl[:,0],fl[:,1],s=20, c='r',linewidths=4)
    ax1.plot(fl[0:7,0],fl[0:7,1], c="tab:red", linewidth=3 , label='Generated')
    ax1.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:red", linewidth=3 )
    ax1.plot(fl[12:17,0],fl[12:17,1], c="tab:red", linewidth=3 )
    ax1.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:red", linewidth=3)

    fl = ref  
    ax1.scatter(fl[:,0],fl[:,1],s=20, c='b',linewidths=4)
    ax1.plot(fl[0:7,0],fl[0:7,1], c="tab:cyan", linewidth=3 , label='Reference')
    ax1.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:cyan", linewidth=3 )
    ax1.plot(fl[12:17,0],fl[12:17,1], c="tab:cyan", linewidth=3 )
    ax1.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:cyan", linewidth=3)

    ax1.legend(loc="upper left")
    ax1.invert_yaxis()

    
    fl = pred

    ax2.scatter(fl[:,0],fl[:,1],s=20, c='r',linewidths=4)
    ax2.plot(fl[0:7,0],fl[0:7,1], c="tab:red", linewidth=3 , label='Generated')
    ax2.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:red", linewidth=3 )
    ax2.plot(fl[12:17,0],fl[12:17,1], c="tab:red", linewidth=3 )
    ax2.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:red", linewidth=3)

    fl = gt 
    ax2.scatter(fl[:,0],fl[:,1],s=20, c='g',linewidths=4)
    ax2.plot(fl[0:7,0],fl[0:7,1], c="tab:green", linewidth=3 , label='Ground Truth')
    ax2.plot(np.append(fl[6:12,0],fl[0,0]),np.append(fl[6:12,1],fl[0,1]), c="tab:green", linewidth=3 )
    ax2.plot(fl[12:17,0],fl[12:17,1], c="tab:green", linewidth=3 )
    ax2.plot(np.append(fl[16:20,0],fl[12,0]),np.append(fl[16:20,1],fl[12,1] ), c="tab:green", linewidth=3)


    ax2.legend(loc="upper left")
    ax2.invert_yaxis()
    
    
    seqlip_fig = ax1.get_figure()

    plt.close(fig)

    return  seqlip_fig

"""

This function is from ***MakeItTalk***

Link: https://github.com/yzhou359/MakeItTalk

"""

# Visualize Facial Landmark (Whole Face)
def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''
    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img




