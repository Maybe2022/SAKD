import torch
import torch.nn.functional as F
import torch.nn as nn

def feature_loss(feature_stu, feature_tea, fun = 'mse', T = 20,Norm_fun=None):
    loss_all = 0
    for i in range(len(feature_stu)):
        if fun == 'mse':
            loss_all += F.mse_loss(feature_stu[i], feature_tea[i].detach())
        elif fun == 'l1':
            loss_all += F.l1_loss(feature_stu[i],feature_tea[i].detach())
        elif fun == 'kl':
            loss_all += KL_loss(feature_stu[i], feature_tea[i].detach(),T)
        elif fun == 'norm':
            loss_all = Norm_fun(feature_stu,feature_tea)
            return loss_all
        elif fun == "mag":
            loss_all += F.l1_loss(torch.norm(feature_stu[i]),torch.norm(feature_tea[i]))
    return loss_all

def logits_loss(outputs,  teacher_outputs, T = 1 ):
    """
    loss function for Knowledge Distillation (KD)
    """
    D_KL = F.kl_div(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1),reduction='batchmean') * (T * T)
    return D_KL