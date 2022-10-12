'''
Detect OOD samples with CLF
'''

import argparse
import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.mixture import GaussianMixture as GMM

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader

from models import get_clf
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds

# OOD samples have larger weight
def get_msp_weight(data_loader, clf):
    clf.eval()

    msp_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

            prob = torch.softmax(logit, dim=1)
            msp_weight.extend(torch.max(prob, dim=1)[0].tolist())

    return [1.0 - msp for msp in msp_weight]

# OOD samples have larger weight
def get_abs_weight(data_loader, clf):
    clf.eval()

    abs_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

            prob = torch.softmax(logit, dim=1)
            abs_weight.extend(prob[:, -1].tolist())

    return abs_weight

# OOD samples have larger weight
def get_energy_weight(data_loader, clf):
    clf.eval()
    
    energy_weight = []

    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = clf(data)
            energy_weight.extend((-torch.logsumexp(logit, dim=1)).tolist())
    
    return energy_weight

# OOD samples have larger weight
def get_binary_weight(data_loader, clf):
    clf.eval()

    binary_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            _, _, energy_logit = clf(data, ret_feat=True, ret_el=True)
            # energy_prob = torch.max(torch.softmax(energy_logit, dim=1), dim=1)[0].tolist()
            energy_prob = torch.sigmoid(energy_logit).tolist()
            binary_weight.extend(energy_prob)
    
    return binary_weight

weight_dic = {
    'msp': get_msp_weight,
    'abs': get_abs_weight,
    'energy': get_energy_weight,
    'binary': get_binary_weight
}

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

def visualize_weight(data_loader_id, data_loader_ood, clf):
    
    plt.clf()
    plt.figure(figsize=(100, 100), dpi=100)

    for i in range(100):
        clf_path = Path(args.output_dir) / (str(i+1) + '.pth')
        # clf_path = Path(args.output_dir) / (str(i) + '.pth')

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

        get_weight = weight_dic[args.weighting]
        weight_id = get_weight(data_loader_id, clf)
        weight_ood = get_weight(data_loader_ood, clf)
        
        ax = plt.subplot(10, 10, i+1)
        # plt.subplot(10, 10, i+1)
        
        weight_id = np.asarray(weight_id)
        weight_ood = np.asarray(weight_ood)

        # true weights distribution
        # bin_counts_id, bins_id = np.histogram(weight_id, bins=100)
        # bin_probs_id = bin_counts_id / len(weight_id)
        bin_counts, bins = np.histogram(weight_ood, bins=100)
        bin_probs = bin_counts / len(weight_ood)

        # get the sigmoid prob for bins
        points = np.array([bins[:-1], bin_probs]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # print(segments[0])
        c_norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap='viridis', norm=c_norm)

        lc_colors = torch.sigmoid(clf.module.binary_linear.weight.detach().cpu() * np.array(bins[:-1])).squeeze()
        lc.set_array(lc_colors)
        line = ax.add_collection(lc)
        plt.colorbar(line, ax=ax)
        
        ax.set_xlim(min(bins), max(bins))
        ax.set_ylim(0.0, 0.20)

        sep_line = True
        for bin_ in bins[::-1][1:]:
            if sep_line:
                binary_ = torch.sigmoid(clf.module.binary_linear.weight.detach().cpu() * np.array(bin_)).squeeze().item()
                if  binary_ < 0.99:
                    plt.axvline(x=bin_, ls="-", c="green")
                    sep_line = False
        
        # plt.plot(bins_id[:-1], bin_probs_id, color='g', label='ID Probs')
        plt.plot(bins[:-1], bin_probs, color='r', label='OOD Probs')

        gmm = GMM(n_components=3).fit(weight_ood.reshape(-1, 1))
        m_g = gmm.means_
        c_g  = gmm.covariances_
        w_g = gmm.weights_

        g_seq = np.argsort(m_g, axis=0).squeeze()
        
        m1 = float(m_g[g_seq[0]][0])
        s1 = np.sqrt(float(c_g[g_seq[0]][0][0]))
        w1 = w_g[g_seq[0]]

        m2 = float(m_g[g_seq[1]][0])
        s2 = np.sqrt(float(c_g[g_seq[1]][0][0]))
        w2 = w_g[g_seq[1]]

        m3 = float(m_g[g_seq[2]][0])
        s3 = np.sqrt(float(c_g[g_seq[2]][0][0]))
        w3 = w_g[g_seq[2]]

        estimated_bin_probs_c1 = norm.cdf(bins[1:], m1, s1) - norm.cdf(bins[:-1], m1, s1)
        estimated_bin_probs_c2 = norm.cdf(bins[1:], m2, s2) - norm.cdf(bins[:-1], m2, s2)
        estimated_bin_probs_c3 = norm.cdf(bins[1:], m3, s3) - norm.cdf(bins[:-1], m3, s3)
        estimated_bin_probs_g = estimated_bin_probs_c1 * w1 + estimated_bin_probs_c2 * w2 +  estimated_bin_probs_c3 * w3

        estimated_bin_probs_c1 = [float(estimated_bin_prob_c1) / sum(estimated_bin_probs_c1) for estimated_bin_prob_c1 in estimated_bin_probs_c1] # normalize
        estimated_bin_probs_c2 = [float(estimated_bin_prob_c2) / sum(estimated_bin_probs_c2) for estimated_bin_prob_c2 in estimated_bin_probs_c2] # normalize
        estimated_bin_probs_c3 = [float(estimated_bin_prob_c3) / sum(estimated_bin_probs_c3) for estimated_bin_prob_c3 in estimated_bin_probs_c3] # normalize
        # estimated_bin_probs_g2_adj = [float(estimated_bin_prob_g2_adj) / sum(estimated_bin_probs_g2_adj) for estimated_bin_prob_g2_adj in estimated_bin_probs_g2_adj] # normalize
        estimated_bin_probs_g = [float(estimated_bin_prob_g) / sum(estimated_bin_probs_g) for estimated_bin_prob_g in estimated_bin_probs_g] # normalize

        # plt.plot(bins[:-1], estimated_bin_probs_g1, color='orange', label='OOD GMM-1')
        plt.plot(bins[:-1], estimated_bin_probs_c1, color='k', linestyle='dashed', alpha=0.5, label='OOD GMM-1')
        plt.plot(bins[:-1], estimated_bin_probs_c2, color='b', linestyle='dashed', alpha=0.5, label='OOD GMM-2')
        plt.plot(bins[:-1], estimated_bin_probs_c3, color='g', linestyle='dashed', alpha=0.5, label='OOD GMM-3')
        # plt.plot(bins[:-1], estimated_bin_probs_g2_adj, color='r', linestyle='dashed', alpha=0.5, label='OOD GMM-2 Adj')
        plt.plot(bins[:-1], estimated_bin_probs_g, color='r', alpha=0.5, label='OOD GMM')

        plt.title(str(i))
    
    # save the figure
    fig_path = Path('./imgs') / args.fig_name
    plt.savefig(str(fig_path))

def main(args):

    # _, std = get_ds_info(args.id, 'mean_and_std')
    test_trf_id = get_ds_trf(args.id, 'test')
    test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    
    test_trf_ood = get_ood_trf(args.id, 'tiny_images', 'test')
    test_all_set_ood = get_ds(root=args.data_dir, ds_name='tiny_images', split='wo_cifar', transform=test_trf_ood)
    indices_sampled_ood = torch.randperm(len(test_all_set_ood))[:int(args.sampled_ood_size_factor * len(test_set_id))].tolist()
    test_set_ood = Subset(test_all_set_ood, indices_sampled_ood)
    test_loader_ood = DataLoader(test_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)

    # load CLF
    num_classes = len(get_ds_info(args.id, 'classes'))
    if args.weighting == 'abs':
        clf = get_clf(args.arch, num_classes+1, args.include_binary)
    elif args.weighting in ['msp', 'energy', 'binary']:
        clf = get_clf(args.arch, num_classes, args.include_binary)
    else:
        raise RuntimeError('<<< Invalid score: '.format(args.score))
    
    clf = nn.DataParallel(clf)
    visualize_weight(test_loader_id, test_loader_ood, clf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialize detection')
    parser.add_argument('--data_dir', type=str, default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--weighting', type=str, default='msp', choices=['msp', 'abs', 'energy', 'binary'])
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--prefetch', type=int, default=16)
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--include_binary', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--sampled_ood_size_factor', type=int, default=5)
    parser.add_argument('--fig_name', type=str, default='test.png')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)