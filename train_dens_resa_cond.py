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
from trainers import get_trainer
from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def get_cond_dens_weight(data_loader, clf, num_classes, sample_mean, precision):
    clf.eval()
    weights = []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            _, penul_feat = clf(data, ret_feat=True)

        # compute class conditional density
        maha_score = 0
        for j in range(num_classes):
            category_sample_mean = sample_mean[j]
            zero_f = penul_feat - category_sample_mean
            
            maha_dis = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if j == 0:
                maha_score = maha_dis.view(-1, 1)
            else:
                maha_score = torch.cat((maha_score, maha_dis.view(-1, 1)), dim=1)
            
        # add to list
        maha_score = torch.sqrt(maha_score) # shape [n * K]
        weights.extend(maha_score.tolist())
    
    weights = np.asarray(weights)
    # print(weights.shape)
    return weights # shape [N * K]

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

    exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, args.training, 'dens_cond', 'r_'+str(args.id_ratio)])
    
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

    trainer = get_trainer(args.training)
    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75, 90], 0.1)

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

        cat_mean, precision = sample_estimator(train_loader_id_test, clf, num_classes)
        weights_candidate_ood = get_cond_dens_weight(train_candidate_loader_ood_test, clf, num_classes, cat_mean, precision)

        # get the dividing point
        weights_id = get_cond_dens_weight(train_loader_id_test, clf, num_classes, cat_mean, precision)

        idxs_sampled = []
        sampled_cond_size = int(args.sampled_ood_size_factor * len(train_set_id_test) / num_classes)

        # sort then quantile ascending by category
        spt_dic = {}
        for i in range(num_classes):
            floor = np.sort(weights_id[:, i])[max(0, int(args.id_ratio * len(train_set_id_test))-1)]

            weights_cond_candidate_ood_sorted = np.sort(weights_candidate_ood[:, i])
            idxs_cond_sorted = np.argsort(weights_candidate_ood[:, i])
            spt = np.searchsorted(weights_cond_candidate_ood_sorted, floor)

            spt_dic[i] = round(100. * spt / len(weights_candidate_ood), 2)
            spt = min(spt, len(weights_candidate_ood) - sampled_cond_size - 1)
            idxs_sampled.extend(idxs_cond_sorted[spt:spt + sampled_cond_size])
        
        # print the corrsponding quantile
        print(spt_dic)

        indices_sampled_ood = [indices_candidate_ood[idx_sampled] for idx_sampled in idxs_sampled]
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
    parser.add_argument('--training', type=str, default='abs', choices=['abs'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--id_ratio', type=float, default=0.95)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='ckpts')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_ood', type=int, default=3072)
    parser.add_argument('--candidate_ood_size', type=int, default=2 ** 20)
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=16, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)