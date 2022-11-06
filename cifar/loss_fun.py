import torch
import torch.nn.functional as F
import torch.nn as nn


def KL_loss(feature_stu, feature_tea,T = 1 ):
    B = feature_stu.shape[0]
    feature_stu = feature_stu.reshape(B, -1)
    feature_tea = feature_tea.reshape(B, -1)
    loss = F.kl_div(F.log_softmax(feature_stu / T, dim=1), F.softmax(feature_tea / T, dim=1),reduction='batchmean') * T * T
    return loss


# def norm(x):
#     B,C,H,W = x.shape
#     return torch.norm(x)/(B * C * H * W)


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

def stage_feature_loss(feature_stu, feature_tea,layer = [2,2,2,2]):
    loss_all = 0

    for i in range(len(feature_tea)):
        tmp_loss = 0
        for j in range(layer[i]):
            tmp_loss += F.mse_loss(feature_stu[sum(layer[:i]) + j],feature_tea[i].detach())
        loss_all += tmp_loss / layer[i]
    return loss_all


class randn_loss(nn.Module):
    def __init__(self , dim = [64,128,256,512],layer = [2,2,2,2]):
        super(randn_loss, self).__init__()
        self.bn = nn.ModuleList()
        for channel in dim:
            self.tmp = nn.BatchNorm2d(channel, affine=False)
            # self.tmp = nn.InstanceNorm2d(channel,affine=False)
            self.tmp.requires_grad_(requires_grad=False)
            self.bn.append(self.tmp)
        self.layer = layer

    def forward(self,fea_stu,fea_tea):
        loss_all = 0
        for i in range(len(self.layer)):
            for j in range(self.layer[i]):
                loss_all += F.mse_loss(self.bn[i](fea_stu[sum(self.layer[:i]) + j]),self.bn[i](fea_tea[sum(self.layer[:i]) + j]).detach())
        return loss_all


class Norm_loss(nn.Module):
    def __init__(self,dim = [64,128,256,512],layer = [1,1,1,1]):
        super(Norm_loss, self).__init__()
        self.bn = nn.ModuleList()
        for channel in dim:
            self.tmp = nn.BatchNorm2d(channel,affine=False)
            # self.tmp = nn.InstanceNorm2d(channel,affine=False)
            self.tmp.requires_grad_(requires_grad=False)
            self.bn.append(self.tmp)
        self.layer = layer

    def forward(self,fea_stu,fea_tea):
        loss = 0
        h = 0

        for i in range(len(self.layer)):
            for j in range(self.layer[i]):
                f1 = self.bn[i](fea_stu[h + j])
                f2 = self.bn[i](fea_tea[h + j])
                loss += F.mse_loss(f1,f2)
            h += self.layer[i]
        return loss



def feature_loss_kl(feature_stu, feature_tea,T = 1):
    B = feature_stu[0].shape[0]
    loss_all = 0
    for i in range(len(feature_stu)):
        loss_all += F.kl_div(F.log_softmax(feature_stu[i].reshape(B,-1) / T, dim=1),F.softmax(feature_tea[i].reshape(B,-1).detach() / T, dim=1),reduction='batchmean') * (T * T)
    return loss_all



def logits_loss(outputs,  teacher_outputs, T = 1 ):
    """
    loss function for Knowledge Distillation (KD)
    """
    D_KL = F.kl_div(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1),reduction='batchmean') * (T * T)
    return D_KL

def divide(x,T = 4):
    B,C,H,W = x.shape
    return x.reshape(T,B//T,C,H,W)
def divide_loss(feature):
    loss_all = 0
    for fea in feature:
        loss = 0
        fea = divide(fea)
        tea = torch.mean(fea,dim=0)
        for f in fea:
            loss += F.mse_loss(f,tea)
        loss_all += loss / 4
    return loss_all



"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class GenSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(GenSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        '''
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
        '''
        if self.contrast_mode == 'all':  # anchor+contrast @ anchor+contrast
            # anchor_labels = torch.cat(labels, dim=0).float()
            # contrast_labels = anchor_labels
            T = features.shape[1]
            labels = labels.unsqueeze(1)
            labels = labels.repeat(1,T, 1)
            anchor_labels = labels.reshape(-1,labels.shape[-1])
            contrast_labels = anchor_labels

            features = features.view(features.shape[0], features.shape[1], -1)
            anchor_features = features.reshape(-1,features.shape[-1])
            contrast_features = anchor_features

            # anchor_features = torch.cat(features, dim=0)
            # contrast_features = anchor_features
        elif self.contrast_mode == 'one':  # anchor @ contrast
            anchor_labels = labels[0].float()
            contrast_labels = labels[1].float()

            anchor_features = features[0]
            contrast_features = features[1]

        # 1. compute similarities among targets
        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True)  # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True)  # [contrast_N, 1]

        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = torch.mm(anchor_labels, contrast_labels.T) / deno  # cosine similarity: [anchor_N, contrast_N]

        logits_mask = torch.ones_like(mask)
        if self.contrast_mode == 'all':
            logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # 2. compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_features, contrast_features.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


# def fuse(feature_snn,feature_cnn):
#     T,B,C,H,W = feature_snn.shape
#     # index = torch.randint(low=0,high=T,size=(1,))
#     out = torch.sum(feature_snn[:T-1]) + feature_cnn
#     return out / T

# x = [1,2,3,4]
# print(sum(x[:0]))
# print(sum(x[4:]))

def time_mask( feature_s,feature_t , mask_rate = 0.5 ):
    mask = torch.rand((1,feature_s.shape[1],1,1),device=feature_s.device)
    mask = torch.where(mask < mask_rate,0,1).to(feature_s.device)
    feature_s = torch.mul(feature_s,mask)
    feature_t = torch.mul(feature_t,1.0 - mask)
    return feature_s,feature_t

class Fuse_loss(nn.Module):
    def __init__(self, dim=[64, 128, 256, 512]):
        super(Fuse_loss, self).__init__()
        self.encoder_s = nn.ModuleList([])
        for channel in dim:
            self.encoder_s.append(nn.Sequential(
                nn.Conv2d(channel,channel,1,1,0,bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                nn.Conv2d(channel,channel,1,1,0,bias=False)
            ))

        self.encoder_t = nn.ModuleList([])
        for channel in dim:
            self.encoder_t.append(nn.Sequential(
                nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
            ))

        self.decoder_s = nn.ModuleList([])
        for channel in dim:
            tmp = nn.Sequential(
                nn.Conv2d(channel,channel,1,1,0,bias=False),
                nn.ReLU(),
                nn.Conv2d(channel,channel,1,1,0,bias=False)
            )
            self.decoder_s.append(tmp)


    def forward(self,feature_s,feature_t):
        loss = 0
        for i in range(len(feature_s)):
            fea_s , fea_t = time_mask(feature_s[i],feature_t[i])
            fea_s_finnal = self.decoder_s[i](self.encoder_s[i](fea_s) + self.encoder_t[i](fea_t))
            loss += F.mse_loss(fea_s_finnal,feature_t[i])
        return loss




class contrast_loss(nn.Module):
    def __init__(self,dim = [64,128,256,512],layer = [1,1,1,1]):
        super(contrast_loss, self).__init__()
        self.contrast = nn.ModuleList()
        for channel in dim:
            self.tmp = linear_average(channel)
            self.contrast.append(self.tmp)
        self.layer = layer

    def forward(self,fea_stu,fea_tea):
        features = []
        for i in range(len(fea_stu)):
            f1 = fea_tea[i].unsqueeze(0)
            fea = torch.cat([fea_stu[i],f1],dim=0)
            features.append(self.contrast[i](fea))
        return features


class linear_average(nn.Module):
    def __init__(self,channel):
        super(linear_average, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channel,channel)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.avg(x)
        x = torch.flatten(x,2)
        return F.normalize(self.relu(self.linear(x)),dim=2)

class deepsup(nn.Module):
    def __init__(self,num_classes = 100):
        super(deepsup, self).__init__()
        self.conv1_tea = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(64,512))
        self.conv2_tea = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(128,512))
        self.conv3_tea = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(256,512))
        # self.conv4_tea = nn.Linear(512, 512)
        self.project_tea = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU()
        )
        self.fc_tea = nn.Linear(512, num_classes)


        self.conv1_stu = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(64,512))
        self.conv2_stu = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(128,512))
        self.conv3_stu = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(256,512))
        # self.conv4_stu = nn.Linear(512, 512)
        self.project_stu = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc_stu = nn.Linear(512,num_classes)



    def forward(self,fea_stu,fea_tea):
        # fea_stu = self.project_stu(self.conv1_stu(fea_stu[0]) + self.conv2_stu(fea_stu[1]) + self.conv3_stu(fea_stu[2]))
        # fea_tea = self.project_stu(self.conv1_tea(fea_tea[0]) + self.conv2_tea(fea_tea[1]) + self.conv3_tea(fea_tea[2]))
        #
        # loss_fea = F.mse_loss(fea_stu,fea_tea.detach())
        # logits_stu = self.fc_stu(fea_stu)
        # logits_tea = self.fc_tea(fea_stu)

        fea_stu[0] , fea_stu[1] , fea_stu[2] = self.conv1_stu(fea_stu[0]) , self.conv2_stu(fea_stu[1]) , self.conv3_stu(fea_stu[2])

        fea_tea[0] , fea_tea[1] , fea_tea[2] = self.conv1_tea(fea_tea[0]) , self.conv2_tea(fea_tea[1]) , self.conv3_tea(fea_tea[2])
        loss_fea = F.mse_loss(fea_stu[0],fea_tea[0].detach()) + F.mse_loss(fea_stu[1],fea_tea[1].detach()) + F.mse_loss(fea_stu[2],fea_tea[2].detach())
        logits_stu = self.project_stu(self.project_stu(fea_stu[0] + fea_stu[1] + fea_stu[2]))
        logits_tea = self.project_tea(self.project_tea(fea_tea[0] + fea_tea[1] + fea_tea[2]))
        return logits_stu,logits_tea,loss_fea





