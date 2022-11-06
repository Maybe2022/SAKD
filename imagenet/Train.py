


import argparse
import os
os.system('wandb login f1ff739b893fd48fb835c7cb39cbe54968b34c44')
import random
import shutil
import time
import warnings
import math
from math import cos, pi

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from spikingjelly.clock_driven import neuron, functional, surrogate
from models import ImageNet_cnn, ImageNet_snn
from loss_kd import feature_loss,logits_loss
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epoch", type=int, default=120)
parser.add_argument("--model", type=str, default='18')
parser.add_argument("--beta", type=float, default=10.)
parser.add_argument("--logit", action='store_true', default=False)
parser.add_argument("--feature", action='store_true', default=False)
parser.add_argument("--both", action='store_true', default=False)
parser.add_argument("--c", type=float, default=10.)
parser.add_argument("--load_weight" ,action='store_true', default=False)
parser.add_argument("--fea_epochs", type=int, default=10)
parser.add_argument("--names", type=str, default='imagenet_train_18')
args = parser.parse_args()



func_dict = {
    '18': [ImageNet_snn.resnet18_, ImageNet_cnn.resnet18],
    '34': [ImageNet_snn.resnet34_, ImageNet_cnn.resnet34],
    '50': [ImageNet_snn.resnet50_, ImageNet_cnn.resnet50],
    '101': [ImageNet_snn.resnet101_, ImageNet_cnn.resnet101],
}



name = 'res' + args.model
data_path = '/gdata/image2012/'  # todo: input your data path
batch_size = args.batch
learning_rate = 1e-3
epochs = args.epoch
distributed = True
print_freq = 5000
warmup = False
warm_epoch = 5
torch.backends.cudnn.benchmark = True
scalar = torch.cuda.amp.GradScaler()



def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    if args.local_rank == 0:
        wandb.init(project="distil_snn", name=args.names, group="ImageNet")
    best_acc1 = .0

    dist.init_process_group(backend='nccl')
    set_seed(1000)
    # create model
    print("=> creating model '{}'".format(name))

    model_fun , teacher_fun = func_dict[args.model]
    model = model_fun(num_classes=1000, T=4)
    teacher = teacher_fun(num_classes=1000)
    print("=> created model '{}'".format(name))
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    teacher.cuda(local_rank)

    teacher.load_state_dict(torch.load('./checkpoints/resnet' + args.model + '_timm.pth'), strict=False)

    if args.load_weight:
        model.load_state_dict(torch.load('./checkpoints/resnet' + args.model + '_timm.pth'), strict=False)
        print("load_weight_success")
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[local_rank])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False, sampler=val_sampler)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=epochs, lr_min=0, lr_max=learning_rate,
                             warmup=warmup)

        # train for one epoch
        acc1_train = train(train_loader, model,teacher, criterion, optimizer, epoch, local_rank,
              args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.local_rank == 0:
            wandb.log({"test_acc": acc1, "train_acc": acc1_train})
            save_checkpoint(
                {'epoch': epoch + 1,
                'arch': name,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                }, is_best)


def train(train_loader, model, teacher ,criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            with torch.cuda.amp.autocast():
                if args.kd:
                    with torch.no_grad():
                        lables, feature_tea = teacher(images)
                output, fea_stu = model(images)

                loss_ce = criterion(output, target)
                loss_feature = feature_loss(fea_stu, feature_tea)
                loss_logit = logits_loss(output, lables)

                if args.both:
                    if i > args.fea_epochs:
                        loss = loss_ce + loss_feature * 0.01 + loss_logit * args.c
                    else:
                        loss = loss_ce + loss_feature * args.beta + loss_logit * args.c
                elif args.feature:
                    if i > args.fea_epochs:
                        loss = loss_ce + loss_feature * 0.01
                    else:
                        loss = loss_ce + loss_feature * args.beta
                elif args.logit:
                    loss = loss_ce + loss_logit * args.c
                else:
                    loss = loss_ce


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        functional.reset_net(model)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print('* loss@1 {losses.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(losses=losses, top1=top1,
                                                                                      top5=top5))
    return top1.avg


def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))
            functional.reset_net(model)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('* loss@1 {losses.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(losses=losses, top1=top1,
                                                                                          top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint34_res.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best34_res.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, current_epoch, max_epoch=epochs, lr_min=0, lr_max=learning_rate, warmup=True):
    warmup_epoch = warm_epoch if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()