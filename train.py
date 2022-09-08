"""
Training on ID data for classification
"""

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import get_clf
from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ds

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

# Training function
def train(data_loader, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit, target)

        # backward
        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if linear_scheduler is not None:
            linear_scheduler.step()
        
        # evaluate
        _, pred = logit.max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader),
        'cla_acc': 100. * correct / total
    }

# Test function
def test(data_loader, net):
    net.eval()

    total, correct = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for sample in data_loader:
            data = sample['data'].cuda()
            target = sample['label'].cuda()

            logit = net(data)
            total_loss += F.cross_entropy(logit, target).item()

            _, pred = logit.max(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader),
        'cla_acc': 100. * correct / total
    }

def main(args):
    # initialize random seed
    init_seeds(args.seed)

    # specify output dir
    # exp_path = Path(args.output_dir) / args.dataset / '-'.join([args.arch, args.clf_type, 'ce', args.scheduler])
    exp_path = Path(args.output_dir) / args.dataset / '-'.join([args.arch, 'ce', args.scheduler])
    print('>>> Output Dir {}'.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # record log
    setup_logger(str(exp_path), 'console.log')

    # init dataset & dataloader
    train_trf = get_ds_trf(args.dataset, stage='train')
    test_trf = get_ds_trf(args.dataset, stage='test')

    train_set = get_ds(root=args.data_dir, ds_name=args.dataset, split='train', transform=train_trf)
    test_set = get_ds(root=args.data_dir, ds_name=args.dataset, split='test', transform=test_trf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('>>> Dataset {}'.format(args.dataset))

    # instantiate net
    num_classes = len(get_ds_info(args.dataset, 'classes'))
    print('>>> CLF {}'.format(args.arch))
    clf = get_clf(args.arch, num_classes)
    clf = nn.DataParallel(clf)

    # move CLF to gpu device
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
    cudnn.benchmark = True

    parameters, linear_parameters = [], []
    for name, parameter in clf.named_parameters():
        if name == 'module.linear.weight' or name == 'module.linear.bias':
            linear_parameters.append(parameter)
        else:
            parameters.append(parameter)
    
    lr_stones = [int(args.epochs * float(lr_stone)) for lr_stone in args.lr_stones]
    optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    linear_optimizer = torch.optim.SGD(linear_parameters, lr=args.lr, weight_decay=args.linear_weight_decay, momentum=args.momentum, nesterov=True)

    if args.scheduler == 'multistep':
        print('LR: {:.2f} - WD: {:.5f} - LWD: {:.5f} - Mom: {:.2f} - Nes: True - LMS: {}'.format(args.lr, args.weight_decay, args.linear_weight_decay, args.momentum, args.lr_stones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_stones, gamma=0.1)
        linear_scheduler = torch.optim.lr_scheduler.MultiStepLR(linear_optimizer, milestones=lr_stones, gamma=0.1)
    elif args.scheduler == 'lambda':
        print('LR: {:.2f} - WD: {:.5f} - LWD: {:.5f} - Mom: {:.2f} - Nes: True'.format(args.lr, args.weight_decay, args.linear_weight_decay, args.momentum))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,
                1e-6 / args.lr
            )
        )

        linear_scheduler = torch.optim.lr_scheduler.LambdaLR(
            linear_optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,
                1e-6 / args.lr
            )
        )
    else:
        raise RuntimeError('<<< Invalid scheduler: {}'.format(args.scheduler))

    begin_time = time.time()
    start_epoch = 1
    cla_acc, best_acc = 0.0, 0.0

    for epoch in range(start_epoch, args.epochs+1):

        if args.scheduler == 'multistep':
            train(train_loader, clf, optimizer, linear_optimizer)
            scheduler.step()
            linear_scheduler.step()
        elif args.scheduler == 'lambda':
            train(train_loader, clf, optimizer, linear_optimizer, scheduler, linear_scheduler)
        else:
            raise RuntimeError('<<< Invalid scheduler: {}'.format(args.scheduler))
        val_metrics = test(test_loader, clf)
        cla_acc = val_metrics['cla_acc']
        clf_best = cla_acc > best_acc
        
        if clf_best:
            best_acc = cla_acc

            cla_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(clf.state_dict()),
                'cla_acc': best_acc
            }
        
        print(
            "---> Epoch {:4d} | Time {:5d}s".format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )
    
    # ------------------------------------ Trainig done, save model ------------------------------------
    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': copy.deepcopy(clf.state_dict()),
        'cla_acc': cla_acc
    }, str(exp_path / 'cla_last.pth'))

    cla_best_path = exp_path / 'cla_best.pth'
    torch.save(cla_best_state, str(cla_best_path))
    print('---> Best classify acc: {:.4f}%'.format(best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CLF')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialize training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='/data/cv')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--linear_weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='lambda', choices=['lambda', 'multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9]) # specify for multistep scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prefetch', type=int, default=16, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)