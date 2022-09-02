'''
Tuning or training with auxiliary OOD training data by classification resampling
'''

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

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

# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

# absent class confidences
def get_abs_softmax_weight(data_loader, clf):
    clf.eval()
    abs_scores = []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

        prob = torch.softmax(logit, dim=1)
        abs_scores.extend(prob[:, -1].tolist())
    
    return torch.tensor(abs_scores)

def get_cond_negative_softmax_weight(data_loader, clf):
    clf.eval()
    max_probs, full_probs = [], []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)
        
        prob = torch.softmax(logit, dim=1)
        full_probs.extend(prob.tolist())
        prob_max = torch.max(prob, dim=1)[0]
        max_probs.extend(prob_max.tolist())
    
    return -1.0 * torch.tensor(max_probs), -1.0 * torch.tensor(full_probs)

def get_cond_softmax_weight(data_loader, clf):
    clf.eval()
    max_probs, full_probs = [], []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            logit = clf(data)

        prob = torch.softmax(logit, dim=1)
        full_probs.extend(prob.tolist())
        prob_max = torch.max(prob, dim=1)[0]
        max_probs.extend(prob_max.tolist())

    return np.asarray(max_probs), np.asarray(full_probs)

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
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader),
        'cla_acc': 100. * correct / total
    }

def main(args):
    init_seeds(args.seed)

    # exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, args.clf_type, args.training, args.scheduler, 'conf', 'q_'+str(args.ood_quantile)])
    exp_path = Path(args.output_dir) / (args.id + '-' + args.ood) / '-'.join([args.arch, args.clf_type, args.training, args.scheduler, 'conf'])

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
    if args.training == 'uni':
        clf = get_clf(args.arch, num_classes, args.clf_type)
    elif args.training == 'abs':
        clf = get_clf(args.arch, num_classes+1, args.clf_type)
    clf = nn.DataParallel(clf)

    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # training parameters
    parameters, linear_parameters = [], []
    for name, parameter in clf.named_parameters():
        if name == 'module.linear.weight' or name == 'module.linear.bias':
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

    train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
    test_trf_ood = get_ood_trf(args.id, args.ood, 'test')
    train_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=train_trf_ood)
    train_all_set_ood_test = get_ds(root=args.data_dir, ds_name=args.ood, split='wo_cifar', transform=test_trf_ood)

    plt.figure(figsize=(100, 100), dpi=100)
    for epoch in range(start_epoch, args.epochs+1):
        
        indices_candidate_ood = torch.randperm(len(train_all_set_ood))[:args.candidate_ood_size].tolist()
        train_candidate_set_ood_test = Subset(train_all_set_ood_test, indices_candidate_ood)
        train_candidate_loader_ood_test = DataLoader(train_candidate_set_ood_test, batch_size=args.batch_size_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        if args.training == 'uni':
            # weights_candidate_ood, _ = get_cond_negative_softmax_weight(train_candidate_loader_ood_test, clf)
            weights_candidate_ood, _ = get_cond_softmax_weight(train_candidate_loader_ood_test, clf)
        elif args.training == 'abs':
            weights_candidate_ood = get_abs_softmax_weight(train_candidate_loader_ood_test, clf)
        
        # estimate the normal distribution & prob distribution
        # gmm_c1 = GMM(n_components=1).fit(weights_candidate_ood.reshape(-1, 1))
        # gmm_c2 = GMM(n_components=2).fit(weights_candidate_ood.reshape(-1, 1))

        # weights_candidate_ood = np.asarray(weights_candidate_ood)
        gmm1 = GMM(n_components=1).fit(weights_candidate_ood.reshape(-1, 1))
        m_g1 = float(gmm1.means_[0][0])
        s_g1 = np.sqrt(float(gmm1.covariances_[0][0][0]))

        gmm2 = GMM(n_components=2).fit(weights_candidate_ood.reshape(-1, 1))
        # gmm = GMM(n_components=1).fit(score_ood.reshape(-1, 1))
        m_g2 = gmm2.means_
        c_g2  = gmm2.covariances_
        weights = gmm2.weights_
        # print(weights)

        m1_g2 = float(m_g2[0][0])
        s1_g2 = np.sqrt(float(c_g2[0][0][0]))

        m2_g2 = float(m_g2[1][0])
        s2_g2 = np.sqrt(float(c_g2[1][0][0]))

        x_axis_ = np.arange(1.0 / num_classes - 0.01, 1.0, 0.01)
        x_axis = np.arange(1.0 / num_classes, 1.0 + 0.01, 0.01)
        
        # estimation for true conf distribution
        # GMM 1
        estimated_bin_probs_g1 = norm.cdf(x_axis, m_g1, s_g1) - norm.cdf(x_axis_, m_g1, s_g1)
        estimated_bin_probs_g1 = [float(estimated_bin_prob_g1) / sum(estimated_bin_probs_g1[1:]) for estimated_bin_prob_g1 in estimated_bin_probs_g1[1:]] # normalize

        # GMM 2
        estimated_bin_probs_g2 = (norm.cdf(x_axis, m1_g2, s1_g2) - norm.cdf(x_axis_, m1_g2, s1_g2)) * weights[0] + (norm.cdf(x_axis, m2_g2, s2_g2) - norm.cdf(x_axis_, m2_g2, s2_g2)) * weights[1]
        estimated_bin_probs_g2 = [float(estimated_bin_prob_g2) / sum(estimated_bin_probs_g2[1:]) for estimated_bin_prob_g2 in estimated_bin_probs_g2[1:]] # normalize
        
        # GMM 2 with adjusting weights
        estimated_bin_probs_adj = (norm.cdf(x_axis, m1_g2, s1_g2) - norm.cdf(x_axis_, m1_g2, s1_g2)) * 0.5 + (norm.cdf(x_axis, m2_g2, s2_g2) - norm.cdf(x_axis_, m2_g2, s2_g2)) * 0.5
        estimated_bin_probs_adj = [float(estimated_bin_prob_adj) / sum(estimated_bin_probs_adj[1:]) for estimated_bin_prob_adj in estimated_bin_probs_adj[1:]] # normalize

        # mean = gmm.means_
        # covs  = gmm.covariances_
        # weights = gmm.weights_

        # without weights
        # mean = (float(gmm_c2.means_[0][0]) + float(gmm_c1.means_[0][0])) / 2.0
        # std = (np.sqrt(float(gmm_c2.covariances_[0][0][0])) + np.sqrt(float(gmm_c1.covariances_[0][0][0]))) / 2.0

        # mean = np.mean(weights_candidate_ood)
        # std = np.std(weights_candidate_ood)
        # bins_ = np.arange(1.0 / num_classes - 0.01, 1.0, 0.01)
        # bins = np.arange(1.0 / num_classes, 1.0 + 0.01, 0.01)
        # estimate target data confidence distribution
        # estimated_bin_probs = (norm.cdf(bins, mean, std) - norm.cdf(bins_, mean, std)) * weights[0]
        # estimated_bin_probs = norm.cdf(bins, mean, std) - norm.cdf(bins_, mean, std)
        
        # target_bin_probs = [float(estimated_bin_prob) / sum(estimated_bin_probs[1:]) for estimated_bin_prob in estimated_bin_probs[1:]]
        # target_bin_probs = estimated_bin_probs_adj
        target_bin_probs = estimated_bin_probs_g2
        target_bin_counts = [int(args.sampled_ood_size_factor * len(train_set_id) * target_bin_prob) for target_bin_prob in target_bin_probs]
        
        # sort confidence into corresponding bins
        bin_labels = np.digitize(weights_candidate_ood, x_axis)
        bin_counts, _ = np.histogram(weights_candidate_ood, x_axis)
        bin_probs = bin_counts / len(weights_candidate_ood)

        idxs_sampled = []
        for i, target_bin_count in enumerate(target_bin_counts):
            target_bin_idxs = np.where(bin_labels==i+1)[0]
            # print(target_bin_idxs.tolist())
            # print(len(target_bin_idxs.tolist()))
            # exit()
            if len(target_bin_idxs) == 0:
                # choose from the most confident bins instead of random choosing?
                idxs_sampled.extend(random.choices(range(len(bin_labels)), k=target_bin_count))
            else:
                idxs_sampled.extend(random.choices(target_bin_idxs, k=target_bin_count))
        
        # idxs_sorted = np.argsort(weights_candidate_ood)
        # spt = int(args.candidate_ood_size * args.ood_quantile)
        # idxs_sampled = idxs_sorted[spt:spt + args.sampled_ood_size_factor * len(train_set_id)]
        plt.subplot(10, 10, epoch)
        plt.plot(x_axis[:-1], bin_probs, color='g', label='true bin probs')
        plt.plot(x_axis[:-1], target_bin_probs, color='r', label='target bin probs')

        indices_sampled_ood = [indices_candidate_ood[idx_sampled] for idx_sampled in idxs_sampled]
        train_set_ood = Subset(train_all_set_ood, indices_sampled_ood)
        train_loader_ood = DataLoader(train_set_ood, batch_size=args.sampled_ood_size_factor * args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)

        if args.scheduler == 'multistep':
            trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer)
            scheduler.step()
            linear_scheduler.step()
        elif args.scheduler == 'lambda':
            trainer(train_loader_id, train_loader_ood, clf, optimizer, linear_optimizer, scheduler, linear_scheduler)
        else:
            raise RuntimeError('<<< Invalid scheduler: {}'.format(args.scheduler))
        val_metrics = test(test_loader, clf, num_classes)
        cla_acc = val_metrics['cla_acc']

        print(
            '---> Epoch {:4d} | Time {:6d}s'.format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )

    # save the figure
    fig_path = Path('./imgs') / args.fig_name
    plt.savefig(str(fig_path))

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
    parser.add_argument('--fig_name', type=str, default='test.png')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--ood', type=str, default='tiny_images')
    parser.add_argument('--training', type=str, default='uni', choices=['abs', 'uni'])
    # parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--clf_type', type=str, default='inner', choices=['euclidean', 'inner'])
    parser.add_argument('--ood_quantile', type=float, default=0.125)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--linear_weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['lambda', 'multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9])
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