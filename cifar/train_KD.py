
# -*- coding: utf-8 -*-
import os
from time import time
os.system('wandb login f1ff739b893fd48fb835c7cb39cbe54968b34c44')
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import torch.nn as nn
from spikingjelly.clock_driven import functional,neuron
import torch
from tools import seed_all,GradualWarmupScheduler
from models import ResNet_cnn,ResNet_snn,ResNet19_cnn,ResNet19_snn
from loss_fun import feature_loss,logits_loss,divide_loss,SupConLoss,Norm_loss,Fuse_loss,contrast_loss,stage_feature_loss,deepsup,randn_loss



seed = 1000
seed_all(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

func_dict = {
    '18': [ResNet_snn.resnet18__, ResNet_cnn.resnet18],
    '34': [ResNet_snn.resnet34__, ResNet_cnn.resnet34],
    '50': [ResNet_snn.resnet50__, ResNet_cnn.resnet50],
    '101': [ResNet_snn.resnet101__, ResNet_cnn.resnet101],
    '152': [ResNet_snn.resnet152__, ResNet_cnn.resnet152],
    '19': [ResNet19_snn.resnet19_,ResNet19_cnn.resnet19]
}
data_path = '/remote-home/share/CIFAR/'
a = 1
feature_b = 100.
logitc_c = 0.
logit_T = 4
kl_T = 1
adamw = False
warm_up = False
lr = 0.1
CIFAR = 100
bathsize = 128
epoch = 200
best_acc = 0.77
sta_time = time()
fun = 'mse'
models = '18'
teacher_model = '18'
names = 'cnn_' + teacher_model + 'teach' + 'snn_' + models
wandb.init(project="distil_snn", name=names, group="Ablation Experiment")

##### 加载模型 ######
s , _ = func_dict[models]
_ , t = func_dict[teacher_model]
teacher = t(num_classes = CIFAR).cuda()
teacher.load_state_dict(torch.load("./model_weight/cnn_"+ teacher_model + "_cifar" + str(CIFAR) +"_baseline.pth"),strict = False)

lenet = s(num_classes = CIFAR).cuda()

n_parameters = sum(p.numel() for p in lenet.parameters() if p.requires_grad)
print('number of params:', n_parameters)
print(lenet)


#######
if adamw:
    learn_rate = lr
    optimer = torch.optim.AdamW(params=lenet.parameters(), lr=learn_rate, betas=(0.9, 0.999), weight_decay=5e-3,eps=1e-3)
else:
    learn_rate = lr
    optimer = torch.optim.SGD(params=lenet.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4,nesterov=True)

###### loss + optim + scheduler ######

loss_fun = nn.CrossEntropyLoss().cuda()
scheduler = CosineAnnealingLR(optimer, T_max=epoch, eta_min=0)
scaler = torch.cuda.amp.GradScaler()
if warm_up:
    scheduler_warmup = GradualWarmupScheduler(optimer,multiplier=1,total_epoch=10,after_scheduler=scheduler)
else:
    scheduler_warmup = None

#######datasets
if CIFAR == 10:
    train_dataset = torchvision.datasets.CIFAR10(root=data_path + 'cifar10', train=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((32, 32), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), download=True)


    test_dataset = torchvision.datasets.CIFAR10(root=data_path + 'cifar10', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]), download=True)

else:
    train_dataset = torchvision.datasets.CIFAR100(root=data_path + 'cifar100', train=True,
      transform=torchvision.transforms.Compose([
          torchvision.transforms.RandomCrop((32, 32), padding=4),
          torchvision.transforms.RandomHorizontalFlip(),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                           (0.2675, 0.2565, 0.2761))
      ]), download=True)


    test_dataset = torchvision.datasets.CIFAR100(root=data_path + 'cifar100', train=False,
         transform=torchvision.transforms.Compose([
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                              (0.2675, 0.2565, 0.2761))
         ]), download=True)

train_data = DataLoader(train_dataset, batch_size=bathsize, shuffle=True, num_workers=4, pin_memory=True)
test_data = DataLoader(test_dataset, batch_size=bathsize, shuffle=False, num_workers=4, pin_memory=True)



def test(model, test_data, data_nums):
    model.eval()
    right = 0
    with torch.no_grad():
        for imgs, label in test_data:
            output = model(imgs.cuda())
            right = (output.argmax(1) == label.cuda()).sum() + right
            functional.reset_net(model)
    return right / data_nums


if __name__ == '__main__':

    for i in range(epoch):

        loss_all = 0
        loss_feature_all = 0
        loss_logits_all = 0
        start_time = time()
        right = 0
        lenet.train()
        teacher.eval()

        for step, (imgs,target) in enumerate(train_data):
                imgs, target = imgs.cuda(),target.cuda()
                # imgs , target = mix(imgs,target)

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        lables , feature_tea = teacher(imgs)

                    output, feature_stu  = lenet(imgs)

                    loss_ce = loss_fun(output,target)

                    loss_feature = feature_loss(feature_stu,feature_tea,fun,T = kl_T)

                    loss_logits = logits_loss(output,lables,logit_T)

                    loss = loss_ce * a + loss_feature * feature_b + loss_logits * logitc_c

                right = (output.argmax(1) == target).sum() + right

                loss_all += loss_ce.item()
                loss_feature_all += loss_feature.item()
                loss_logits_all += loss_logits.item()

                optimer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimer)
                scaler.update()
                functional.reset_net(lenet)

        accuracy1 = right / (len(train_dataset))

        if warm_up:
            scheduler_warmup.step()
        else:
            scheduler.step()
        ###### Test set evaluation
        accuracy = test(lenet,test_data,data_nums=len(test_dataset))

        print("epoch:{} time:{:.0f}  loss_ce :{:.2f} loss_feature :{:.2f} loss_logits :{:.2f} train_acc:{:.4f} tets_acc:{:.4f} lr:{:.4f} eta:{:.2f}".format(i + 1, time() - start_time, loss_all,loss_feature,loss_logits_all,accuracy1, accuracy,optimer.param_groups[0]['lr'], (time() - start_time) * ( epoch - i-1 ) / 3600))

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(lenet.state_dict(), "./model_weight/" + names +".pth")
            print("The model is saved")
            print("best_acc:{}".format(best_acc))

        wandb.log({"test_acc": accuracy, "train_acc": accuracy1, "loss_feature": loss_feature_all,"loss_ce": loss_all, "loss_logits": loss_logits_all , 'epoch': i , 'lr': optimer.param_groups[0]['lr']})

    end_ = time()
    print(end_ - sta_time)
    print("best_acc:{:.4f}".format(best_acc))
