'''
DOS
'''

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader

from models import get_clf
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

# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def main(args):

    init_seeds(args.seed)
    exp_path = Path(args.output_dir) / ('s' + str(args.seed)) / (args.id  + '-' + args.ood) / '-'.join([args.arch, 'abs', args.scheduler, 'b_'+str(args.beta), 'bs_'+str(args.batch_size), 'dos_k_'+str(args.num_cluster)])
    exp_path.mkdir(parents=True, exist_ok=True)

    setup_logger(str(exp_path), 'console.log')
    print('>>> Output dir: {}'.format(str(exp_path)))
    
    train_trf_id = get_ds_trf(args.id, 'train')
    train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
    test_trf = get_ds_trf(args.id, 'test')

    train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
    test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf)
    
    if args.ood in ['ti_300k', 'imagenet_64']:
        train_set_all_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='train', transform=train_trf_ood)

    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True, drop_last=True)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    # the candidate ood idxs
    indices_candidate_ood_epochs = []

    for i in range(args.epochs):
        indices_epoch = np.array(random.sample(range(len(train_set_all_ood)), args.size_candidate_ood))
        indices_candidate_ood_epochs.append(indices_epoch)

    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))
    num_classes = len(get_ds_info(args.id, 'classes'))
    print('>>> CLF: {}'.format(args.arch))
    clf = get_clf(args.arch, num_classes+1)
    
    # move CLF to gpus
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()

    print('Optimizer: LR: {:.2f} - WD: {:.5f} - Mom: {:.2f} - Nes: True'.format(args.lr, args.weight_decay, args.momentum))
    lr_stones = [int(args.epochs * float(lr_stone)) for lr_stone in args.lr_stones]
    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    
    if args.scheduler == 'multistep':
        print('LR: {:.2f} - WD: {:.5f} - Mom: {:.2f} - Nes: True - LMS: {}'.format(args.lr, args.weight_decay, args.momentum, args.lr_stones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_stones, gamma=0.1)
    elif args.scheduler == 'lambda':
        print('LR: {:.2f} - WD: {:.5f} - Mom: {:.2f} - Nes: True'.format(args.lr, args.weight_decay, args.momentum))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
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

    batch_size_candidate_ood = int(args.size_candidate_ood / len(train_set_id) * args.batch_size)
    batch_size_sampled_ood = int(args.size_factor_sampled_ood * args.batch_size)
    spt, ept = args.spt, args.ept

    print(spt, ept)

    for epoch in range(start_epoch, args.epochs+1):

        train_set_candidate_ood = Subset(train_set_all_ood, indices_candidate_ood_epochs[epoch - 1])
        train_loader_candidate_ood = DataLoader(train_set_candidate_ood, batch_size=batch_size_candidate_ood, shuffle=False, num_workers=args.prefetch, pin_memory=True)
        
        epoch_time = time.time()
        
        for sample_id, sample_ood in zip(train_loader_id, train_loader_candidate_ood):

            # select ood in batch
            clf.eval()
            data_batch_candidate_ood = sample_ood['data'].cuda()

            with torch.no_grad():
                logits_batch_candidate_ood, feats_batch_candidate_ood = clf(data_batch_candidate_ood, ret_feat=True)

            prob_ood = torch.softmax(logits_batch_candidate_ood, dim=1)
            weights_batch_candidate_ood = np.array(prob_ood[:, -1].tolist())
            idxs_sorted = np.argsort(weights_batch_candidate_ood)

            # normalize
            repr_batch_candidate_ood = np.array(F.normalize(feats_batch_candidate_ood.cpu(), dim=-1))

            # clustering
            k = args.num_cluster
            kmeans = KMeans(n_clusters=args.num_cluster, n_init=args.n_init).fit(repr_batch_candidate_ood)
            clus_candidate_ood = kmeans.labels_

            idxs_sampled = []

            # --- sub-cluster ---
            if k > batch_size_sampled_ood:
                # if number of cluster larger than ood batch size, then choose 1 ood sample from each cluster
                sampled_cluster_size = 1
            else:
                sampled_cluster_size = int(batch_size_sampled_ood / k)

            for i in range(min(k, batch_size_sampled_ood)):

                valid_idxs = np.where(clus_candidate_ood == i)[0]
                
                if len(valid_idxs) <= sampled_cluster_size:
                    idxs_sampled.extend(valid_idxs)
                else:
                    idxs_valid_sorted = np.argsort(weights_batch_candidate_ood[valid_idxs])
                    idxs_sampled.extend(valid_idxs[idxs_valid_sorted[:sampled_cluster_size]])

            # fill the empty: remove the already sampled, then randomly complete the sampled
            idxs_sampled.extend(random.sample(list(set(idxs_sorted) - set(idxs_sampled)), k=batch_size_sampled_ood - len(idxs_sampled)))
            data_ood = data_batch_candidate_ood[idxs_sampled]
            
            # OE with absent class in batch
            num_classes = len(train_loader_id.dataset.classes)
            clf.train()

            total, correct, total_loss = 0, 0, 0.0

            num_id = sample_id['data'].size(0)
            num_ood = data_ood.size(0)

            data_id = sample_id['data'].cuda()
            data = torch.cat([data_id, data_ood], dim=0)
            target_id = sample_id['label'].cuda()
            target_ood = (torch.ones(num_ood) * num_classes).long().cuda()

            # forward
            logit = clf(data)
            loss = F.cross_entropy(logit[:num_id], target_id)
            loss += args.beta * F.cross_entropy(logit[num_id:], target_ood)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if args.scheduler == 'lambda':
                scheduler.step()

            # evaluate
            _, pred = logit[:num_id, :num_classes].max(dim=1)
            with torch.no_grad():
                total_loss += loss.item()
                correct += pred.eq(target_id).sum().item()
                total += num_id

        if args.scheduler == 'multistep':
            scheduler.step()
        
        print('Epoch Time: ', time.time() - epoch_time)
        
        # average on sample
        print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(train_loader_id), 100. * correct / total))
        
        val_metrics = test(test_loader_id, clf, num_classes)
        cla_acc = val_metrics['cla_acc']

        print(
            '---> Epoch {:4d} | Time {:6d}s'.format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )

        if epoch % args.save_freq == 0 or epoch >= 90:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(clf.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'scheduler': copy.deepcopy(scheduler.state_dict()),
                'cla_acc': cla_acc
            }, str(exp_path / (str(epoch)+'.pth')))

    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': copy.deepcopy(clf.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'cla_acc': cla_acc
    }, str(exp_path / 'cla_last.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sea')
    parser.add_argument('--seed', default=1, type=int, help='seed for init training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='../datasets')
    parser.add_argument('--id', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--ood', type=str, default='ti_300k', choices=['ti_300k', 'imagenet_64'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='tuning')
    parser.add_argument('--arch', type=str, default='densenet101', choices=['densenet101', 'wrn40'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['lambda', 'multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=101)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2 ** 6) # 64
    parser.add_argument('--size_candidate_ood', type=int, default=300000)
    parser.add_argument('--size_factor_sampled_ood', type=int, default=1)

    parser.add_argument('--num_cluster', type=int, default=64) # 192: 24(8) -> 64: 8
    parser.add_argument('--n_init', type=int, default=3)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)
