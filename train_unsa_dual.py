'''
Tuning or training with auxiliary OOD training data by classification undersampling
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
from scores import get_weight

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def test(data_loader, clf, num_classes):
    clf.eval()

    total, correct = 0, 0
    total_loss = 0.0

    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()

        with torch.no_grad():
            # forward
            if clf.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)
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

    exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, args.training, 'b_'+str(args.beta), args.scheduler, 'dual_'+str(args.ood_ratio)])
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
    if args.training in ['uni', 'energy']:
        clf = get_clf(args.arch, num_classes)
    elif args.training == 'abs':
        clf = get_clf(args.arch, num_classes+1)
    elif args.training == 'binary':
        clf = get_clf(args.arch, num_classes, include_binary=True)
    
    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
    clf.apply(weights_init)

    # training parameters
    parameters, linear_parameters = [], []
    for name, parameter in clf.named_parameters():
        
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

    indices_sampled_ood = []
    all_indices_sampled_ood = set()
    
    for epoch in range(start_epoch, args.epochs+1):
        
        init_seeds(epoch_seeds[epoch])

        # candidate
        indices_candidate_ood = torch.randperm(len(train_all_set_ood))[:args.candidate_ood_size].tolist()
        print('ICO:', indices_candidate_ood[:10])

        # sampled
        test_candidate_set_ood = Subset(test_all_set_ood, indices_candidate_ood)
        test_candidate_loader_ood = DataLoader(test_candidate_set_ood, batch_size=args.batch_size_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        weights_candidate_ood = get_weight(test_candidate_loader_ood, clf, args.weighting)
        idxs_sorted = np.argsort(weights_candidate_ood)
        ept = int(args.candidate_ood_size * args.ood_ratio)
        idxs_sampled = random.sample(idxs_sorted[:ept], k=int(args.sampled_ood_size_factor * len(train_set_id)))
        indices_sampled_ood = [indices_candidate_ood[idx_sampled] for idx_sampled in idxs_sampled]
        print('ISO:', indices_sampled_ood[:10])

        # calculate proximity
        print('Proximity:', np.mean(weights_candidate_ood[idxs_sampled]))
        print('Diversity:', 1.0 - len(set(indices_sampled_ood) & all_indices_sampled_ood) / len(indices_sampled_ood))
        all_indices_sampled_ood.update(indices_sampled_ood)

        # training
        train_set_ood = Subset(train_all_set_ood, indices_sampled_ood)
        train_loader_ood = DataLoader(train_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        if args.training == 'binary':
            if epoch <= 10:
                pos_w = 0.5
            else:
                pos_w = 1.0
        
        if args.scheduler == 'multistep':
            if args.training == 'binary':
                trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer, beta=args.beta, pos_w=pos_w)
            else:
                trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer, beta=args.beta)
            scheduler.step()
            linear_scheduler.step()
        elif args.scheduler == 'lambda':
            if args.training == 'binary':
                trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer, scheduler, linear_scheduler, beta=args.beta, pos_w=pos_w)
            else:
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
    parser.add_argument('--weighting', type=str, default='abs', choices=['msp', 'abs', 'energy', 'binary'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='esd')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--linear_weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='lambda', choices=['lambda', 'multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ood_ratio', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_ood', type=int, default=3072)
    parser.add_argument('--candidate_ood_size', type=int, default=2 ** 20)
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)