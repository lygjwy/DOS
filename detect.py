'''
Detect OOD samples with CLF
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
# import sklearn.covariance

import torch
import torch.nn.functional as F
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
            msp_score.extend(torch.max(prob, dim=1))

    return msp_score

score_dic = {
    'msp': get_msp_score
}

def main(args):
    output_path = Path(args.output_dir) / args.id / args.output_sub_dir
    print('>>> Result dir: {}'.format(output_path))
    output_path.mkdir(parents=True, exist_ok=True)

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
    clf = get_clf(args.arch, num_classes)
    clf_path = Path(args.pretrain)

    if clf_path.is_file():
        clf_state = torch.load(str(clf_path))
        cla_acc = clf_state['cla_acc']
        clf.load_state_dict(clf_state['state_dict'])
        print('>>> load classifier from {} (classifiication acc {:.4f}%)'.format(str(clf_path), cla_acc))
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
    result_dic_list = []

    score_id = get_score(test_loader_id, clf)
    label_id = np.ones(len(score_id))

    for test_loader_ood in test_loader_oods:
        result_dic = {'name': test_loader_ood.dataset.name}

        score_ood = get_score(test_loader_ood, clf)
        label_ood = np.zeros(len(score_ood))

        # OOD detection
        score = np.concatenate([score_id, score_ood])
        label = np.concatenate([label_id, label_ood])

        result_dic['fpr_at_tpr'], result_dic['auroc'], result_dic['aupr_in'], result_dic['aupr_out'] = compute_all_metrics(score, label, verbose=False)
        result_dic_list.append(result_dic)
        
        print('---> [ID: {:7s} - OOD: {:9s}] [auroc: {:3.3f}%, aupr_in: {:3.3f}%, aupr_out: {:3.3f}%, fpr@95tpr: {:3.3f}%]'.format(
            test_loader_id.dataset.name, test_loader_ood.dataset.name, 100. * result_dic['auroc'], 100. * result_dic['aupr_in'], 100. * result_dic['aupr_out'], 100. * result_dic['fpr_at_tpr']))

    # save result
    result = pd.DataFrame(result_dic_list)
    result_path = output_path / (args.scores + '.csv')
    result.to_csv(str(result_path), index=False, header=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--data_dir', type=str, default='/data/cv')
    parser.add_argument('--output_dir', help='dir to store log', default='results')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='tmp')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'lsunc', 'dtd', 'places365_10k', 'cifar100', 'tinc', 'lsunr', 'tinr', 'isun'])
    parser.add_argument('--score', type=str, default='msp')
    # parser.add_argument('--temperature', type=int, default=1000)
    # parser.add_argument('--magnitude', type=float, default=0.0014)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--pretrain', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)