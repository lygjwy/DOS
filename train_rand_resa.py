'''
Tuning or training with auxiliary OOD training data by random resampling
'''

import copy
from tabnanny import verbose
import time
import random
import argparse
import numpy as np
from pathlib import Path
from functools import partial

import torch
from torch import nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader

from models import get_clf
from trainers import get_trainer
from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def test(data_loader, net, num_classes):
    net.eval()

    total, correct = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for sample in data_loader:
            data = sample['data'].cuda()
            target = sample['label'].cuda()

            # forward
            logit = net(data)
            total_loss += F.cross_entropy(logit, target).item()

            _, pred = logit[:, :num_classes].max(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader.dataset), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader.dataset),
        'cla_acc': 100. * correct / total
    }

def main(args):
    init_seeds(args.seed)

    exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, args.training, 'rand'])
    print('>>> Output dir: {}'.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)

    setup_logger(str(exp_path), 'console.log')

    train_trf_id = get_ds_trf(args.id, 'train')
    train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
    test_trf = get_ds_trf(args.id, 'test')

    train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
    train_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=train_trf_ood)
    test_set = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf)

    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))
    num_classes = len(get_ds_info(args.id, 'classes'))
    print('>>> CLF: {}'.format(args.arch))
    if args.training == 'uni':
        clf = get_clf(args.arch, num_classes)
    elif args.training == 'abs':
        clf = get_clf(args.arch, num_classes+1)
    clf = nn.DataParallel(clf)

    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # get trainer
    trainer = get_trainer(args.training)
    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75, 90], 0.1)

    begin_time = time.time()
    start_epoch = 1
    cla_acc = 0.0

    for epoch in range(start_epoch, args.epochs+1):

        # resampling auxiliary OOD training samples
        indices_sampled_ood = torch.randperm(len(train_all_set_ood))[:args.sampled_ood_size_factor * len(train_set_id)].tolist()
        train_set_ood = Subset(train_all_set_ood, indices_sampled_ood)
        train_loader_ood = DataLoader(train_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
        
        trainer(train_loader_id, train_loader_ood, clf, optimizer)
        scheduler.step()
        val_metrics  = test(test_loader, clf, num_classes)
        cla_acc = val_metrics['cla_acc']

        print(
            '---> Epoch {:4d} | Time {:6d}s'.format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )

    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': copy.deepcopy(clf.state_dict()),
        'cla_acc': cla_acc
    }, str(exp_path / 'cla_last.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Outlier Exposure')
    parser.add_argument('--seed', default=42, type=int, help='seed for init training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--ood', type=str, default='tiny_images')
    parser.add_argument('--training', type=str, default='abs', choices=['uni', 'abs'])
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='ckpts')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=16, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)