'''
Detect OOD samples with CLF
'''

import argparse
import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
# import sklearn.covariance

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader

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

def get_logit_score(data_loader, clf):
    clf.eval()
    
    logit_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)
            
            max_logit = torch.max(logit, dim=1)[0]
            logit_score.extend(max_logit.tolist())
    
    return logit_score

def get_energy_score(data_loader, clf, temperature=1.0):
    clf.eval()
    
    energy_score = []

    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = clf(data)
            energy_score.extend((temperature * torch.logsumexp(logit / temperature, dim=1)).tolist())
    
    return energy_score

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
    'logit': get_logit_score,
    'energy': get_energy_score
}

def main(args):

    # _, std = get_ds_info(args.id, 'mean_and_std')
    test_trf_id = get_ds_trf(args.id, 'test')
    test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    
    test_trf_ood = get_ood_trf(args.id, 'tiny_images', 'test')
    test_all_set_ood = get_ds(root=args.data_dir, ds_name='tiny_images', split='wo_cifar', transform=test_trf_ood)
    indices_sampled_ood = torch.randperm(len(test_all_set_ood))[:args.sampled_ood_size_factor * len(test_set_id)].tolist()
    test_set_ood = Subset(test_all_set_ood, indices_sampled_ood)
    test_loader_ood = DataLoader(test_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)

    # load CLF
    num_classes = len(get_ds_info(args.id, 'classes'))
    clf = get_clf(args.arch, num_classes, args.clf_type)
    clf = nn.DataParallel(clf)
    
    # visualize the confidence distribution
    plt.figure(figsize=(100, 100), dpi=100)

    # for model_name in args.models:
    for i in range(100):
        clf_path = Path(args.output_dir) / (str(i) + '.pth')

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

        get_score = score_dic[args.score]
        score_id = get_score(test_loader_id, clf)
        label_id = np.ones(len(score_id))
    
        score_ood = get_score(test_loader_ood, clf)
        label_ood = np.zeros(len(score_ood))

        # OOD detection
        score = np.concatenate([score_id, score_ood])
        label = np.concatenate([label_id, label_ood])

        # plot the histgrams
        bins = 100
        plt.subplot(10, 10, i+1)
        # negative log of original score
        # score_id = np.log(score_id)
        plt.hist(score_id, bins, density=True, color='g', label='id', alpha=0.5)
        thr_95 = np.sort(score_id)[int(len(score_id) * 0.05)]
        plt.axvline(thr_95)
        
        # score_ood = np.log(score_ood)
        score_ood = np.asarray(score_ood)
        gmm = GMM(n_components=1).fit(score_ood.reshape(-1, 1))
        mean = gmm.means_
        covs  = gmm.covariances_
        weights = gmm.weights_

        x_axis = np.arange(0.01, 1.0, 0.01)
        # x_axis = np.arange(-5.0, 0.0, 0.01)
        y_axis0 = norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0])))*weights[0] # 1nd gaussian
        # y_axis1 = norm.pdf(x_axis, float(mean[1][0]), np.sqrt(float(covs[1][0][0])))*weights[1] # 2nd gaussian
        plt.plot(x_axis, y_axis0, lw=3, c='C0')
        # plt.plot(x_axis, y_axis1, lw=3, c='C1')
        
        plt.hist(score_ood, bins, density=True, color='r', label='ood', alpha=0.5)
        plt.title(str(i))

        compute_all_metrics(score, label, True)
        
    # save the figure
    plt.savefig(args.fig_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialize detection')
    parser.add_argument('--data_dir', type=str, default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--score', type=str, default='msp', choices=['msp', 'logit', 'energy'])
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--prefetch', type=int, default=16)
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--clf_type', type=str, default='inner', choices=['inner', 'euclidean'])
    parser.add_argument('--output_dir', type=str, default='./ckpts')
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--fig_name', type=str, default='test.png')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)