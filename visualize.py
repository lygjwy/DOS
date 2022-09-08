'''
Detect OOD samples with CLF
'''

import argparse
import numpy as np
from pathlib import Path
import sklearn.covariance
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.mixture import GaussianMixture as GMM


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

def get_energy_score(data_loader, clf, temperature=1.0):
    clf.eval()
    
    energy_score = []
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = clf(data)
            energy_score.extend((temperature * -torch.logsumexp(logit / temperature, dim=1)).tolist())
    
    return energy_score

def get_binary_score(data_loader, clf):
    clf.eval()

    binary_score = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            _, _, energy_logit = clf(data, ret_feat=True, ret_el=True)
            # energy_prob = torch.max(torch.softmax(energy_logit, dim=1), dim=1)[0].tolist()
            energy_prob = torch.sigmoid(energy_logit).tolist()
            binary_score.extend(energy_prob)
    
    return binary_score

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

def visualize_score(data_loader_id, data_loader_ood, clf):
    num_classes = len(data_loader_id.dataset.classes)
    
    plt.clf()
    plt.figure(figsize=(100, 100), dpi=100)

    test_trf_id = get_ds_trf(args.id, 'test')
    train_set_id_test = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=test_trf_id)
    train_loader_id_test = DataLoader(train_set_id_test, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

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

        # get_score = score_dic[args.score]
        if args.score == 'maha':
            cat_mean, precision = sample_estimator(train_loader_id_test, clf, num_classes)
            get_score = partial(
                score_dic['maha'],
                num_classes=num_classes, 
                sample_mean=cat_mean, 
                precision=precision
            )
        else:
            get_score = score_dic[args.score]

        score_id = get_score(data_loader_id, clf)
        score_ood = get_score(data_loader_ood, clf)
        
        plt.subplot(10, 10, i+1)
        
        score_id = np.asarray(score_id)
        score_ood = np.asarray(score_ood)
        # gmm1 = GMM(n_components=1).fit(score.reshape(-1, 1))
        # m_g1 = float(gmm1.means_[0][0])
        # s_g1 = np.sqrt(float(gmm1.covariances_[0][0][0]))

        # gmm2 = GMM(n_components=2).fit(score.reshape(-1, 1))
        # m_g2 = gmm2.means_
        # c_g2  = gmm2.covariances_
        # weights = gmm2.weights_
        # print(weights)

        # m1_g2 = float(m_g2[0][0])
        # s1_g2 = np.sqrt(float(c_g2[0][0][0]))

        # m2_g2 = float(m_g2[1][0])
        # s2_g2 = np.sqrt(float(c_g2[1][0][0]))

        # x_axis_ = np.arange(1.0 / num_classes - 0.01, 1.0, 0.01)
        # x_axis = np.arange(1.0 / num_classes, 1.0 + 0.01, 0.01)

        
        # estimation for true conf distribution
        # GMM 1
        # estimated_bin_probs_g1 = norm.cdf(x_axis, m_g1, s_g1) - norm.cdf(x_axis_, m_g1, s_g1)
        # estimated_bin_probs_g1 = [float(estimated_bin_prob_g1) / sum(estimated_bin_probs_g1[1:]) for estimated_bin_prob_g1 in estimated_bin_probs_g1[1:]] # normalize

        # GMM 2
        # estimated_bin_probs_g2 = (norm.cdf(x_axis, m1_g2, s1_g2) - norm.cdf(x_axis_, m1_g2, s1_g2)) * weights[0] + (norm.cdf(x_axis, m2_g2, s2_g2) - norm.cdf(x_axis_, m2_g2, s2_g2)) * weights[1]
        # estimated_bin_probs_g2 = [float(estimated_bin_prob_g2) / sum(estimated_bin_probs_g2[1:]) for estimated_bin_prob_g2 in estimated_bin_probs_g2[1:]] # normalize
        
        # GMM 2 with adjusting weights
        # estimated_bin_probs = (norm.cdf(x_axis, m1_g2, s1_g2) - norm.cdf(x_axis_, m1_g2, s1_g2)) * 0.5 + (norm.cdf(x_axis, m2_g2, s2_g2) - norm.cdf(x_axis_, m2_g2, s2_g2)) * 0.5
        # estimated_bin_probs = [float(estimated_bin_prob) / sum(estimated_bin_probs[1:]) for estimated_bin_prob in estimated_bin_probs[1:]] # normalize

        # bin_counts, _ = np.histogram(score, x_axis)
        # bin_probs = bin_counts / len(score)
        
        # plt.plot(x_axis[:-1], bin_probs, color='g', label='true bin probs')
        
        # plt.plot(x_axis[:-1], estimated_bin_probs, color='orange', label='target bin probs')
        # plt.plot(x_axis[:-1], estimated_bin_probs_g1, color='b', label='c1 bin probs')
        # plt.plot(x_axis[:-1], estimated_bin_probs_g2, color='yellow', label='c2 bin probs')
        # x_axis = np.arange(0.0, 1.0+0.01, 0.01)
        
        # const = 100.0
        # const = 50.0
        # score = (const + score) / const # normalization

        # bin_counts, _ = np.histogram(score, x_axis)
        # bin_probs = bin_counts / len(score)
        
        # plt.plot(x_axis[:-1], bin_probs, color='orange', label='target bin probs')
        plt.hist(score_id, bins=100, color='g', alpha=0.5)
        plt.hist(score_ood, bins=100, color='r', alpha=0.5)

        plt.title(str(i))
        
    # save the figure
    fig_path = Path('./imgs') / args.fig_name
    plt.savefig(str(fig_path))

score_dic = {
    'msp': get_msp_score,
    'logit': get_logit_score,
    'maha': get_mahalanobis_score,
    'abs': get_abs_score,
    'energy': get_energy_score,
    'binary': get_binary_score
}

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
    num_classes = len(get_ds_info(args.id, 'classes'))
    if args.score == 'abs':
        clf = get_clf(args.arch, num_classes+1, args.include_binary)
    elif args.score in ['maha', 'logit', 'energy', 'msp', 'binary']:
        clf = get_clf(args.arch, num_classes, args.include_binary)
    else:
        raise RuntimeError('<<< Invalid score: '.format(args.score))
    
    clf = nn.DataParallel(clf)
    # visualize
    # visualize_trend(test_loader_ood, clf)
    # visualize_score(test_loader_id, test_loader_ood, clf, num_classes)
    visualize_score(test_loader_id, test_loader_ood, clf)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialize detection')
    parser.add_argument('--data_dir', type=str, default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--score', type=str, default='msp', choices=['msp', 'logit', 'maha', 'energy', 'abs', 'binary'])
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--prefetch', type=int, default=16)
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--include_binary', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./ckpts')
    parser.add_argument('--sampled_ood_size_factor', type=int, default=2)
    parser.add_argument('--fig_name', type=str, default='test.png')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)