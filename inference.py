'''
Evaluate the CLF inference speed on TinyImages
'''

import time
import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import get_clf
from datasets import get_ds_info, get_ds

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

def main(args):
    
    # load data
    mean, std = get_ds_info(args.id, 'mean_and_std')
    test_trf_ood = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='test', transform=test_trf_ood)
    test_loader_ood = DataLoader(dataset=test_set_ood, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

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
        clf = nn.DataParallel(clf)
        clf.cuda()
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    begin_time = time.time()
    get_msp_score(test_loader_ood, clf)
    print('Time: {:d}s'.format(time.time() - begin_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on auxiliary OOD training data')
    parser.add_argument('--seed', type=int, default=42, help='seed for inference')
    parser.add_argument('--data_dir', type=str, default='/data/cv/')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--ood', type=str, default='tiny_images')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=0)
    parser.add_argument('--arch', type=str, default='wrn40')
    parser.add_argument('--pretrain', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)