'''
Tuning or training with auxiliary OOD training data by classification resampling
'''

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path
from sklearn.covariance import EmpiricalCovariance

import torch
from torch import nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader

from models import get_clf
from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_conf_weight(data_loader, clf):
    clf.eval()
    weights = []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)
        
        prob = torch.softmax(logit, dim=1)
        # prob_max, cat = torch.max(prob, dim=1)
        weights.extend(prob[:, -1].tolist())
    
    return torch.tensor(weights)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def train(data_loader_id, data_loader_ood, net, num_classes, optimizer, scheduler):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)
        num_ood = sample_ood['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target_id = sample_id['label'].cuda()
        target_ood =  (torch.ones(num_ood) * num_classes).long().cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target_id)
        loss += 1.0 * F.cross_entropy(logit[num_id:], target_ood)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # evaluate
        _, pred = logit[:num_id].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target_id).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id.dataset), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id.dataset),
        'cla_acc': 100. * correct / total
    }

def test(data_loader, net):
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

            _, pred = logit.max(dim=1)
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

    exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, 'conf', 'q_'+str(args.ood_quantile)])
    
    print('>>> Output dir: {}'.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)

    setup_logger(str(exp_path), 'console.log')

    train_trf_id = get_ds_trf(args.id, 'train')
    test_trf_id = get_ds_trf(args.id, 'test')

    train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
    test_set = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)

    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))

    num_classes = len(get_ds_info(args.id, 'classes'))
    print('>>> CLF: {}'.format(args.arch))
    # absentation class
    clf = get_clf(args.arch, num_classes+1)
    clf = nn.DataParallel(clf)

    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_id),
            1,
            1e-6 / args.lr
        )
    )

    begin_time = time.time()
    start_epoch = 1
    cla_acc = 0.0

    train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
    test_trf_ood = get_ood_trf(args.id, args.ood, 'test')
    train_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=train_trf_ood)
    train_all_set_ood_test = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=test_trf_ood)

    for epoch in range(start_epoch, args.epochs+1):
        indices_candidate_ood = torch.randperm(len(train_all_set_ood))[:args.candidate_ood_size].tolist()
        train_candidate_set_ood_test = Subset(train_all_set_ood_test, indices_candidate_ood)
        train_candidate_loader_ood_test = DataLoader(train_candidate_set_ood_test, batch_size=args.batch_size_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        weights_candidate_ood = get_conf_weight(train_candidate_loader_ood_test, clf)
        # sort then quantile
        idxs_sorted = np.argsort(weights_candidate_ood)
        spt = int(args.candidate_ood_size * args.ood_quantile)
        idxs_sampled = idxs_sorted[spt:spt + args.sampled_ood_size_factor * len(train_set_id)]

        indices_sampled_ood = [indices_candidate_ood[idx_sampled] for idx_sampled in idxs_sampled]
        train_set_ood = Subset(train_all_set_ood, indices_sampled_ood)
        train_loader_ood = DataLoader(train_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)

        train(train_loader_id, train_loader_ood, clf, num_classes, optimizer, scheduler)
        val_metrics  = test(test_loader, clf)
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
    parser.add_argument('--ood_quantile', type=float, default=0.125)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='ckpts')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_ood', type=int, default=3072)
    parser.add_argument('--candidate_ood_size', type=int, default=2 ** 20)
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=16, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)