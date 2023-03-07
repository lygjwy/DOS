'''
Tuning or training with auxiliary OOD training data by classification undersampling
'''

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path

from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

import torch
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

    exp_path = Path(args.output_dir) / ('s' + str(args.seed)) / (args.id + '-' + '_' + args.ood) / '-'.join([args.arch, 't', 'b_'+str(args.beta), args.scheduler, 'k_'+str(args.num_cluster), 'g_'+str(args.num_group)])
    exp_path.mkdir(parents=True, exist_ok=True)

    setup_logger(str(exp_path), 'console.log')
    print('>>> Output dir: {}'.format(str(exp_path)))
    
    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))
    num_classes = len(get_ds_info(args.id, 'classes'))

    train_trf_id = get_ds_trf(args.id, 'train')
    train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
    test_trf_id = get_ds_trf(args.id, 'test')
    test_trf_ood = get_ood_trf(args.id, args.ood, 'test')

    train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
    test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)
    if args.ood == 'tiny_images':
        train_all_set_ood = get_ds(root=args.data_dir, ds_name='tiny_images', split='wo_cifar', transform=train_trf_ood)
        test_all_set_ood = get_ds(root=args.data_dir, ds_name='tiny_images', split='wo_cifar', transform=test_trf_ood)
    elif args.ood in ['ti_300k', 'imagenet_64']:
        train_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='train', transform=train_trf_ood)
        test_all_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='train', transform=test_trf_ood)

    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    # load pretrained model
    extractor = get_clf(args.extractor, 1000)
    extractor_path = Path(args.pretrain)

    if extractor_path.is_file():
        extractor_state = torch.load(str(extractor_path))
        extractor.load_state_dict(extractor_state)
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        extractor.cuda()
    
    # feature clustering
    test_loader_all_ood = DataLoader(test_all_set_ood, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    extractor.eval()
    feats_ood = []
    for sample in test_loader_all_ood:

        data = sample['data'].cuda()
        
        with torch.no_grad():
            _, feat_ood = extractor(data, ret_feat=True)
            feats_ood.extend(feat_ood.tolist())
    
    # grouping by clustering
    k = args.num_cluster
    kmeans = KMeans(n_clusters=args.num_cluster, n_init=10).fit(feats_ood)
    clus_ood = kmeans.labels_

    # get the training data idxs
    ## candidate cluster
    size_clus = []
    idxs_all_clus = []
    for i in range(k):
        idxs_clus = np.where(clus_ood == i)[0]
        size_clus.append(len(idxs_clus))
        idxs_all_clus.append(idxs_clus)
        
    idx_sorted_size_clus = np.argsort(np.array(size_clus))

    idxs_sampled_ood = []
    for i in range(args.num_group):
        size_avg_clus = int(len(train_set_id) / args.num_group)
        if size_clus[idx_sorted_size_clus[i]] > size_avg_clus:
            idxs_sampled_ood.extend(random.sample(idxs_all_clus[idx_sorted_size_clus[i]], k=size_avg_clus))
        else:
            idxs_sampled_ood.extend(random.choices(idxs_all_clus[idx_sorted_size_clus[i]], k=size_avg_clus))

    train_set_ood = Subset(train_all_set_ood, indices=idxs_sampled_ood)
    train_loader_ood = DataLoader(train_set_ood, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    print('>>> CLF: {}'.format(args.arch))
    if args.training in ['uni', 'energy']:
        clf = get_clf(args.arch, num_classes)
    elif args.training == 'abs':
        clf = get_clf(args.arch, num_classes+1)
    
    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
    # clf.apply(weights_init)

    # training parameters
    parameters, linear_parameters = [], []
    for name, parameter in clf.named_parameters():
        if name in ['linear.weight', 'linear.bias']:
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

    for epoch in range(start_epoch, args.epochs+1):

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

        if epoch % args.save_freq == 0:
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
    
    np.save(str(exp_path / 'diversity_scores.npy'), np.array(d_s))

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

    parser = argparse.ArgumentParser(description='sea')
    parser.add_argument('--seed', default=42, type=int, help='seed for init training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='/data/cv')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--ood', type=str, default='ti_300k', choices=['tiny_images', 'ti_300k', 'imagenet_64'])
    parser.add_argument('--training', type=str, default='abs', choices=['uni', 'abs', 'energy'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='esd')
    parser.add_argument('--extractor', type=str, default='wrn40_2', choices=['wrn40_2'])
    parser.add_argument('--pretrain', type=str, default='/home/hzheng/jwy/sea/models/imagenet_wrn_baseline_epoch_99.pt')
    parser.add_argument('--arch', type=str, default='densenet101', choices=['densenet101', 'wrn40_4'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--linear_weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['lambda', 'multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=101)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2 ** 6) # 64
    parser.add_argument('--num_cluster', type=int, default=6)
    parser.add_argument('--num_group', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)