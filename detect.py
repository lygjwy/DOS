'''
Detect OOD samples with CLF
'''

import argparse
import numpy as np
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
# import sklearn.covariance

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import get_clf
from utils import compute_all_metrics
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

def get_msp_score(data_loader, clf):
    clf.eval()

    msp_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

            prob = torch.softmax(logit, dim=1)
            msp_score.extend(torch.max(prob, dim=1)[0].tolist())

    return msp_score

def get_abs_score(data_loader, clf):
    '''
    Probability for absent class
    '''
    clf.eval()

    abs_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

            prob = torch.softmax(logit, dim=1)
            abs_score.extend(prob[:, -1].tolist())

    return [1 - abs for abs in abs_score]

def get_logit_score(data_loader, clf):
    clf.eval()

    logit_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)
            logit_score.extend(torch.max(logit, dim=1)[0].tolist())

    return logit_score

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

def get_mahalanobis_score(data_loader, clf, num_classes, sample_mean, precision):
    '''
    Negative mahalanobis distance to the cloest class center
    '''
    clf.eval()

    nm_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            _, penul_feat = clf(data, ret_feat=True)

        term_gaus = torch.empty(0)
        for j in range(num_classes):
            category_sample_mean = sample_mean[j]
            zero_f = penul_feat - category_sample_mean
            # term_gau = torch.exp(-0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()) # [BATCH,]
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # [BATCH, ]
            if j == 0:
                term_gaus = term_gau.view(-1, 1)
            else:
                term_gaus = torch.cat((term_gaus, term_gau.view(-1, 1)), dim=1)

        nm_score.extend(torch.max(term_gaus, dim=1)[0].tolist())

    return nm_score

def get_energy_score(data_loader, clf, temperature=1.0):
    clf.eval()
    
    energy_score = []

    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = clf(data)
            energy_score.extend((temperature * torch.logsumexp(logit / temperature, dim=1)).tolist())
    
    return energy_score

def get_binary_score(data_loader, clf):
    clf.eval()

    binary_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            _, _, energy_logit = clf(data, ret_feat=True, ret_el=True)
            # energy_prob = torch.max(torch.softmax(energy_logit, dim=1), dim=1)[0].tolist()
            energy_prob = torch.sigmoid(energy_logit).squeeze().tolist()
            binary_score.extend(energy_prob)
    
    return [-1.0 * bs + 1.0 for bs in binary_score]

def get_acc(data_loader, clf, num_classes):
    clf.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for sample in data_loader:
            data = sample['data'].cuda()
            target = sample['label'].cuda()

            logit = clf(data)

            _, pred = logit[:, :num_classes].max(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    print(correct / total * 100.)
    return correct / total * 100.

score_dic = {
    'msp': get_msp_score,
    'abs': get_abs_score,
    'logit': get_logit_score,
    'maha': get_mahalanobis_score,
    'energy': get_energy_score,
    'binary': get_binary_score
}

def main(args):

    # _, std = get_ds_info(args.id, 'mean_and_std')
    test_trf_id = get_ds_trf(args.id, 'test')
    test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    
    test_loader_oods = []
    for ood in args.oods:
        test_trf_ood = get_ood_trf(args.id, ood, 'test')
        test_set_ood = get_ds(root=args.data_dir, ds_name=ood, split='test', transform=test_trf_ood)
        test_loader_oods.append(DataLoader(test_set_ood, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True))

    # load CLF
    num_classes = len(get_ds_info(args.id, 'classes'))
    if args.score == 'abs':
        clf = get_clf(args.arch, num_classes+1, args.include_binary)
    elif args.score in ['maha', 'logit', 'energy', 'msp', 'binary']:
        clf = get_clf(args.arch, num_classes, args.include_binary)
    else:
        raise RuntimeError('<<< Invalid score: '.format(args.score))
    
    clf = nn.DataParallel(clf)
    clf_path = Path(args.pretrain)

    if clf_path.is_file():
        clf_state = torch.load(str(clf_path))
        cla_acc = clf_state['cla_acc']
        clf.load_state_dict(clf_state['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f}%)'.format(str(clf_path), cla_acc))
    else:
        raise RuntimeError('<--- invlaid classifier path: {}'.format(str(clf_path)))

    # move CLF to gpu device
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # get_acc(test_loader_id, clf, num_classes)

    get_score = score_dic[args.score]
    if args.score == 'maha':
        train_set_id_test = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=test_trf_id)
        train_loader_id_test = DataLoader(train_set_id_test, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
        cat_mean, precision = sample_estimator(train_loader_id_test, clf, num_classes)
        get_score = partial(
            score_dic['maha'],
            num_classes=num_classes, 
            sample_mean=cat_mean, 
            precision=precision
        )
    else:
        get_score = score_dic[args.score]
    score_id = get_score(test_loader_id, clf)
    label_id = np.ones(len(score_id))

    # visualize the confidence distribution
    plt.figure(figsize=(10, 10), dpi=100)
    
    ood_names, fprs, aurocs, auprs = [], [], [], []
    for i, test_loader_ood in enumerate(test_loader_oods):
        # result_dic = {'name': test_loader_ood.dataset.name}
        ood_names.append(test_loader_ood.dataset.name)

        score_ood = get_score(test_loader_ood, clf)
        label_ood = np.zeros(len(score_ood))

        # OOD detection
        score = np.concatenate([score_id, score_ood])
        label = np.concatenate([label_id, label_ood])

        # plot the histgrams
        bins = np.linspace(0.0, 1.0, 100)
        plt.subplot(3, 3, i+1)
        plt.hist(score_id, bins, color='g', label='id', alpha=0.5)
        thr_95 = np.sort(score_id)[int(len(score_id) * 0.05)]
        plt.axvline(thr_95, alpha=0.5)
        plt.hist(score_ood, bins, color='r', label='ood', alpha=0.5)
        plt.title(test_loader_ood.dataset.name)

        fpr, auroc, aupr, _ = compute_all_metrics(score, label)
        
        fprs.append(100. * fpr)
        aurocs.append(100. * auroc)
        auprs.append(100. * aupr)

    # save the figure
    plt.savefig(args.fig_name)

    # print results
    print('[ ID: {:7s} - OOD:'.format(args.id), end=' ')
    for ood_name in ood_names:
        print('{:5s}'.format(ood_name), end=' ')
    print(']')

    print('> FPR:  ', end=' ')
    for fpr in fprs:
        print('{:3.3f}'.format(fpr), end=' ')
    print('<')

    print('> AUROC:', end=' ')
    for auroc in aurocs:
        print('{:3.3f}'.format(auroc), end=' ')
    print('<')

    print('> AUPR: ', end=' ')
    for aupr in auprs:
        print('{:3.3f}'.format(aupr), end=' ')
    print('<')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialize detection')
    parser.add_argument('--data_dir', type=str, default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'lsunc', 'dtd', 'places365_10k', 'cifar100', 'tinc', 'lsunr', 'tinr', 'isun'])
    parser.add_argument('--score', type=str, default='msp', choices=['msp', 'abs', 'logit', 'maha', 'energy', 'binary'])
    # parser.add_argument('--temperature', type=int, default=1000)
    # parser.add_argument('--magnitude', type=float, default=0.0014)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--prefetch', type=int, default=16)
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--include_binary', action='store_true')
    parser.add_argument('--pretrain', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--fig_name', type=str, default='test.png')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)