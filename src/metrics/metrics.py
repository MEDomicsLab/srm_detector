"""
    @file:              metrics.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi

    @Creation Date:     06/2024
    @Last modification: 07/2024

    @Description:       This file is used to define the metrics.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def get_metrics(prediction, label):
    ''' 
    Arguments: 
        predicrion --> model output
        label      --> ground trutch
    Returns classification metrics of the model prediction in a dictionary
    '''
    prediction  = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    tp = np.sum((prediction == 1) & (label == 1))
    tn = np.sum((prediction == 0) & (label == 0))
    fp = np.sum((prediction == 1) & (label == 0))
    fn = np.sum((prediction == 0) & (label == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    try:
        auc = roc_auc_score(label, prediction)
    except ValueError:
        auc = None
    fpr, tpr, _ = roc_curve(label, prediction)
    return {'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr}


def binary_accuracy(preds, y):
    ''' 
    Arguments: 
        preds --> model output
        y     --> ground trutch
    Returns the accuracy of the model prediction
    '''
    rounded_preds = torch.round(preds)
    correct       = (rounded_preds == y).float()
    acc           = correct.sum() / len(correct)
    return acc