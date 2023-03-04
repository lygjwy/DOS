'''
300K image subset from 80 million Tiny Images. Images belonging to CIFAR classes, Places or LSUN classes, and images with divisive metadata are removed.
Code adapted from https://github.com/hongxin001/ODNL
'''

import random
import numpy as np
from pathlib import Path
from bisect import bisect_left

from torch.utils.data import Dataset

class TI300K(Dataset):

    def __init__(self, root, transform=None, split='train'):

        self.transform = transform

        data_file_path = Path(root) / 'ti_300k' / split / '300K_random_images.npy'
        self.data = np.load(data_file_path).astype(np.uint8)

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return {
            'data': img
        }
    
    def __len__(self):

        return len(self.data)
