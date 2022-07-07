'''
Tuning or training with auxiliary OOD training data by classification resampling
'''

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path
from functools import partial
from sklearn.covariance import EmpiricalCovariance

import torch
from torch import nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader

from models import get_clf
from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_logit_cond_weight(data_loader, clf):
    clf.eval()
    weights, cats = [], []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

        logit_max, cat = torch.max(logit, dim=1)
        weights.extend(logit_max.tolist())
        cats.extend(cat.tolist())

    return weights, cats

def get_prob_cond_weight(data_loader, clf):
    clf.eval()
    weights, cats = [], []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)
        
        prob = torch.softmax(logit, dim=1)
        prob_max, cat = torch.max(prob, dim=1) 
        weights.extend(prob_max.tolist())
        cats.extend(cat.tolist())
    
    return weights, cats

def sample_estimator(data_loader, clf, num_classes):
    clf.eval()
    group_lasso = EmpiricalCovariance(assume_centered=False)

    num_sample_per_class = np.zeros(num_classes)
    list_features = [0] * num_classes

    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()

        with torch.no_grad():
            _, penulti_feature = clf(data, ret_feat=True)
        
        # construct the sample matrix
        for i in range(target.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                list_features[label] = penulti_feature[i].view(1, -1)
            else:
                list_features[label] = torch.cat((list_features[label], penulti_feature[i].view(1, -1)), 0)
            num_sample_per_class[label] += 1

    category_sample_mean = []
    for j in range(num_classes):
        category_sample_mean.append(torch.mean(list_features[j], 0))

    X = 0
    for j in range(num_classes):
        if j == 0:
            X = list_features[j] - category_sample_mean[j]
        else:
            X = torch.cat((X, list_features[j] - category_sample_mean[j]), 0)
        
    # find inverse
    group_lasso.fit(X.cpu().numpy())
    precision = group_lasso.precision_
    
    return category_sample_mean, torch.from_numpy(precision).float().cuda()

def get_den_cond_weight(data_loader, clf, num_classes, sample_mean, precision):
    clf.eval()
    weights, cats = [], []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            _, penul_feat = clf(data, ret_feat=True)

        # compute class conditional density
        gaussian_score = 0
        for j in range(num_classes):
            category_sample_mean = sample_mean[j]
            zero_f = penul_feat - category_sample_mean
            term_gau = torch.exp(-0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag())
            if j == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
    
        # add to list   
        den_max, cat = torch.max(gaussian_score, dim=1)
        weights.extend(den_max.tolist())
        cats.extend(cat.tolist())
    
    return weights, cats

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def train(data_loader_id, data_loader_ood, net, optimizer, scheduler):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target)
        loss += 0.5 * -(logit[num_id:].mean(dim=1) - torch.logsumexp(logit[num_id:], dim=1)).mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # evaluate
        _, pred = logit[:num_id].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
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

    if args.pretrain is not None:
        pretrain = 'tune'
    else:
        pretrain = 'train'
    
    if args.replacement:
        exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, pretrain, 'weighted', args.weight_type, args.training, 'wr'])
    else:
        exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, pretrain, 'weighted', args.weight_type, args.training, 'wor'])

    print('>>> Output dir: {}'.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)

    setup_logger(str(exp_path), 'console.log')

    train_trf_id = get_ds_trf(args.id, 'train')
    test_trf_id = get_ds_trf(args.id, 'test')

    train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
    train_set_id_test = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=test_trf_id)
    test_set = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)

    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    train_loader_id_test = DataLoader(train_set_id_test, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))

    num_classes = len(get_ds_info(args.id, 'classes'))
    print('>>> CLF: {}'.format(args.arch))
    clf = get_clf(args.arch, num_classes)
    clf = nn.DataParallel(clf)

    if args.pretrain is not None:
        clf_path = Path(args.pretrain)

        if clf_path.is_file():
            clf_state = torch.load(str(clf_path))
            cla_acc = clf_state['cla_acc']
            clf.load_state_dict(clf_state['state_dict'])
            print('>>> load CLF from {} (classifiication acc {:.4f}%)'.format(str(clf_path), cla_acc))
        else:
            raise RuntimeError('<<< invalid classifier path: {}'.format(str(clf_path)))
    
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

    cond_weight_dic = {
        'logit': get_logit_cond_weight,
        'prob': get_prob_cond_weight,
        'density': get_den_cond_weight
    }
    get_cond_weight = cond_weight_dic[args.cond_weight]
    
    if args.training == 'fix':
        # random select 2 ** 24 --> 2 ** 20
        indices_ood = torch.randperm(len(train_all_set_ood))[:2 ** 20].tolist()
        train_candidate_set_ood = Subset(train_all_set_ood, indices_ood)
        train_candidate_set_ood_test = Subset(train_all_set_ood_test, indices_ood)

    for epoch in range(start_epoch, args.epochs+1):

        if args.training == 'var':
            indices_ood = torch.randperm(len(train_all_set_ood))[:2 ** 20].tolist()
            train_candidate_set_ood = Subset(train_all_set_ood, indices_ood)
            train_candidate_set_ood_test = Subset(train_all_set_ood_test, indices_ood)

        train_candidate_loader_ood_test = DataLoader(train_candidate_set_ood_test, batch_size=args.batch_size_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        if args.cond_weight == 'density':
            cat_mean, precision = sample_estimator(train_loader_id_test, clf, num_classes)
            weights_ood, conds_ood = get_cond_weight(train_candidate_loader_ood_test, clf, num_classes, cat_mean, precision)
        else:
            weights_ood, conds_ood = get_cond_weight(train_candidate_loader_ood_test, clf)

        if args.cond_sample:
            # conditional sampling
            indices_sampled_ood = []
            for i in range(num_classes):
                indices_cond_ood = (conds_ood == i).nonzero()
                weights_cond_ood = weights_ood[indices_cond_ood]

                idx_to_index = indices_cond_ood.squeeze()
                rand_idxs = torch.multinomial(weights_cond_ood, 2 * len(train_set_id) / num_classes, args.replacement, generator=None)
                indices_sampled_ood.extend([idx_to_index[rand_idx] for rand_idx in rand_idxs.tolist()])

            train_sampled_set_ood = Subset(train_candidate_set_ood, indices_sampled_ood)
            train_loader_ood = DataLoader(train_sampled_set_ood, batch_size=2*args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
        else:
            # non-conditional sampling
            weighted_train_sampler_ood = WeightedRandomSampler(weights_ood, num_samples=2 * len(train_set_id), replacement=args.replacement)
            train_loader_ood = DataLoader(train_candidate_set_ood, batch_size=2*args.batch_size, sampler=weighted_train_sampler_ood, num_workers=args.prefetch, pin_memory=True)
        
        train(train_loader_id, train_loader_ood, clf, optimizer, scheduler)
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
    parser.add_argument('--training', type=str, default='fix', choices=['fix', 'var'])
    parser.add_argument('--cond_weight', type=str, default='density', choices=['logit', 'prob', 'density'])
    parser.add_argument('--cond_sample', type=str, action='store_true')
    parser.add_argument('--replacement', action='store_true')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--pretrain', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_ood', type=int, default=3072)
    parser.add_argument('--prefetch', type=int, default=16, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)