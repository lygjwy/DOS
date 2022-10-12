'''
Tuning or training with auxiliary OOD training data by classification resampling
'''

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
from torch.utils.data import Subset, DataLoader

from models import get_clf, weights_init
from trainers import get_trainer
from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

# OOD samples have larger weight
def get_energy_weight(data_loader, clf):
    clf.eval()
    
    energy_weight = []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            if args.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)

            energy_weight.extend((-torch.logsumexp(logit, dim=1)).tolist())
    
    return energy_weight

# OOD samples have smaller score
def get_kl_score(data_loader, clf):
    clf.eval()
    
    kl_score = []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            if args.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)
            
            softmax = torch.softmax(logit, dim=1)
        
            uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
            kl_score.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return kl_score

def get_energy_kl(data_loader, clf):
    clf.eval()

    energy_weight = []
    kl_score = []

    if args.parallel:
        print(clf.module.binary_linear.weight.data)
    else:
        print(clf.binary_linear.weight.data)

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            if args.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)
            energy_weight.extend((-torch.logsumexp(logit, dim=1)).tolist())

            softmax = torch.softmax(logit, dim=1)
            uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
            kl_score.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return np.asarray(energy_weight), np.asarray(kl_score)

def get_inter_idxs_sampled(data_loader, clf, total_num, target_num, energy_ratio, kl_ratio):
    energy_ood, kl_ood = get_energy_kl(data_loader, clf)

    energy_sorted_idxs = np.argsort(energy_ood)[:int(total_num * energy_ratio)]
    kl_sorted_idxs = np.argsort(kl_ood)[:int(total_num * kl_ratio)]
    inter_idxs = [idx for idx in energy_sorted_idxs if idx in kl_sorted_idxs]
    print('Inter length:', len(inter_idxs))

    if len(inter_idxs) < target_num:
        # energy_ratio = min(energy_ratio + 0.05, 1.0)
        kl_ratio = min(kl_ratio + 0.1, 1.0)
        print('ER:', energy_ratio, 'KR:', kl_ratio)
        idxs_sampled = get_inter_idxs_sampled(data_loader, clf, total_num, target_num, energy_ratio, kl_ratio)
    else:
        idxs_sampled = random.sample(inter_idxs, k=target_num)
    
    return idxs_sampled

def test(data_loader, net, num_classes):
    net.eval()

    total, correct = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for sample in data_loader:
            data = sample['data'].cuda()
            target = sample['label'].cuda()

            # forward
            if args.include_binary:
                logit, _ = net(data)
            else:
                logit = net(data)
            
            total_loss += F.cross_entropy(logit, target).item()

            _, pred = logit[:, :num_classes].max(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader),
        'cla_acc': 100. * correct / total
    }

def main(args):
    init_seeds(args.seed)

    epoch_seeds = [args.seed]
    for i in range(args.epochs):
        random.seed(epoch_seeds[i])
        epoch_seeds.append(random.randint(1000 * i, 1000 * (i+1)))

    exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, args.training, args.scheduler, 'bidir', 'b_'+str(args.beta), 'er_'+str(args.energy_ratio), 'kr_'+str(args.kl_ratio)])
    exp_path.mkdir(parents=True, exist_ok=True)

    setup_logger(str(exp_path), 'console.log')
    print('>>> Output dir: {}'.format(str(exp_path)))
    
    train_trf_id = get_ds_trf(args.id, 'train')
    train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
    test_trf_id = get_ds_trf(args.id, 'test')
    test_trf_ood = get_ood_trf(args.id, args.ood, 'test')

    train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
    train_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=train_trf_ood)
    test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)
    test_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=test_trf_ood)

    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))
    num_classes = len(get_ds_info(args.id, 'classes'))
    print('>>> CLF: {}'.format(args.arch))
    if args.training in ['uni', 'energy', 'binary']:
        clf = get_clf(args.arch, num_classes, args.include_binary)
    elif args.training == 'abs':
        clf = get_clf(args.arch, num_classes+1, args.include_binary)
    # multi gpus
    if args.parallel:
        clf = nn.DataParallel(clf)
    
    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
    cudnn.benchmark = True
    clf.apply(weights_init)

    # training parameters
    parameters, linear_parameters = [], []
    for name, parameter in clf.named_parameters():
        if args.parallel:
            if name in ['module.linear.weight', 'module.linear.bias', 'module.binary_linear.weight', 'module.binary_linear.bias']:
                linear_parameters.append(parameter)
            else:
                parameters.append(parameter)
        else:
            if name in ['linear.weight', 'linear.bias', 'binary_linear.weight', 'binary_linear.bias']:
                linear_parameters.append(parameter)
            else:
                parameters.append(parameter)

    print('Optimizer: LR: {:.2f} - WD: {:.5f} - LWD: {:.5f} - Mom: {:.2f} - Nes: True'.format(args.lr, args.weight_decay, args.linear_weight_decay, args.momentum))
    trainer = get_trainer(args.training)
    lr_stones = [int(args.epochs * float(lr_stone)) for lr_stone in args.lr_stones]
    optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    linear_optimizer = torch.optim.SGD(linear_parameters, lr=args.lr, weight_decay=args.linear_weight_decay, momentum=args.momentum, nesterov=True)
    
    if args.scheduler == 'multistep':
        print('Scheduler: MultiStepLR - LMS: {}'.format(args.lr_stones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_stones, gamma=0.1)
        linear_scheduler = torch.optim.lr_scheduler.MultiStepLR(linear_optimizer, milestones=lr_stones, gamma=0.1)
    elif args.scheduler == 'lambda':
        print('Scheduler: LambdaLR')
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader_id),
                1,
                1e-6 / args.lr
            )
        )
        linear_scheduler = torch.optim.lr_scheduler.LambdaLR(
            linear_optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader_id),
                1,
                1e-6 / args.lr
            )
        )
    else:
        raise RuntimeError('<<< Invalid scheduler: {}'.format(args.scheduler))
    
    begin_time = time.time()
    start_epoch = 1
    cla_acc = 0.0

    all_indices_sampled_ood = set()
    for epoch in range(start_epoch, args.epochs+1):
        
        init_seeds(epoch_seeds[epoch])

        indices_candidate_ood = torch.randperm(len(train_all_set_ood))[:args.candidate_ood_size].tolist()
        print('ICO:', indices_candidate_ood[:10])
        test_candidate_set_ood = Subset(test_all_set_ood, indices_candidate_ood)
        test_candidate_loader_ood = DataLoader(test_candidate_set_ood, batch_size=args.batch_size_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        # energy_candidate_ood, kl_candidate_ood = get_energy_kl(test_candidate_loader_ood, clf)
        # energy_candidate_ood = np.asarray(get_energy_weight(test_candidate_loader_ood, clf))
        # kl_candidate_ood = np.asarray(get_kl_score(test_candidate_loader_ood, clf))

        # energy_ratio = int(args.candidate_ood_size * args.energy_ratio)
        # kl_ratio = int(args.candidate_ood_size * args.kl_ratio)
        # energy_sorted_idxs = np.argsort(energy_candidate_ood)[:energy_ratio]
        # kl_sorted_idxs = np.argsort(kl_candidate_ood)[:kl_ratio]
        # inter_idxs = [idx for idx in energy_sorted_idxs if idx in kl_sorted_idxs]
        # print('Inter length:', len(inter_idxs))

        # if len(inter_idxs) < (args.sampled_ood_size_factor * len(train_set_id)):

        # else:
        #     # idxs_sampled = torch.randperm(len(inter_idxs))[:args.sampled_ood_size_factor * len(train_set_id)].tolist()
        #     idxs_sampled = random.sample(inter_idxs, k=(args.sampled_ood_size_factor * len(train_set_id)))

        idxs_sampled = get_inter_idxs_sampled(test_candidate_loader_ood, clf, args.candidate_ood_size, args.sampled_ood_size_factor * len(train_set_id), args.energy_ratio, args.kl_ratio)
        indices_sampled_ood = [indices_candidate_ood[idx_sampled] for idx_sampled in idxs_sampled]
        # print('ISO:', indices_sampled_ood[:10])

        # calculate proximity
        test_set_ood = Subset(test_all_set_ood, indices_sampled_ood)
        test_loader_ood = DataLoader(test_set_ood, batch_size=args.batch_size_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)
        weights_ood_test = np.asarray(get_energy_weight(test_loader_ood, clf))
        print('Mean:', np.mean(weights_ood_test))
        print('Median:', np.median(weights_ood_test))
        print('Diversity:', 1.0 - len(set(indices_sampled_ood) & all_indices_sampled_ood) / len(indices_sampled_ood))

        all_indices_sampled_ood.update(indices_sampled_ood)
        train_set_ood = Subset(train_all_set_ood, indices_sampled_ood)
        train_loader_ood = DataLoader(train_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)

        if args.scheduler == 'multistep':
            trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer, beta=args.beta)
            scheduler.step()
            linear_scheduler.step()
        elif args.scheduler == 'lambda':
            trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer, scheduler, linear_scheduler, beta=args.beta)
        else:
            raise RuntimeError('<<< Invalid scheduler: {}'.format(args.scheduler))
        val_metrics = test(test_loader_id, clf, num_classes)
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
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'linear_optimizer': copy.deepcopy(linear_optimizer.state_dict()),
            'scheduler': copy.deepcopy(scheduler.state_dict()),
            'linear_scheduler': copy.deepcopy(linear_scheduler.state_dict()),
            'cla_acc': cla_acc
        }, str(exp_path / (str(epoch)+'.pth')))
    
    # Total sampled imgs number
    print('Total:', len(all_indices_sampled_ood))

    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': copy.deepcopy(clf.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'linear_optimizer': copy.deepcopy(linear_optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'linear_scheduler': copy.deepcopy(linear_scheduler.state_dict()),
        'cla_acc': cla_acc
    }, str(exp_path / 'cla_last.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Outlier Exposure')
    parser.add_argument('--seed', default=42, type=int, help='seed for init training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--ood', type=str, default='tiny_images')
    parser.add_argument('--training', type=str, default='binary', choices=['uni', 'abs', 'energy', 'binary'])
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--include_binary', action='store_true')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='e1')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--linear_weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='lambda', choices=['lambda', 'multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--energy_ratio', type=float, default=0.5)
    parser.add_argument('--kl_ratio', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_ood', type=int, default=3072)
    parser.add_argument('--candidate_ood_size', type=int, default=2 ** 20)
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    
    main(args)