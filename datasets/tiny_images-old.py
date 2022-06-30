'''
80 million Tiny Images Dataset (https://groups.csail.mit.edu/vision/TinyImages/)
Code adapted from from https://github.com/hendrycks/outlier-exposure
'''

import numpy as np
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

class TinyImages(Dataset):

    def __init__(self, root, transform=None, exclude_cifar=True):

        data_dir = Path(root) / 'tiny_images'
        idx_file_path = data_dir / '80mn_cifar_idxs.txt'
        
        self.data_file = open(data_dir / 'tiny_images.bin', 'rb')
        
        self.num = 79302017
        self.offset = 0 # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            # indices in file take the 80mn database to start at 1, hence "- 1"
            with open(idx_file_path, 'r') as idxs:
                cifar_idxs = [int(idx)-1 for idx in idxs]

            # hash table option
            cifar_idxs = set(cifar_idxs)
            self.in_cifar = lambda x: x in cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % self.num

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(self.num)

        img = self._load_image(index)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return {
            'data': img
        }

    def __len__(self):
        return self.num

    def _load_image(self, idx):
        self.data_file.seek(idx * 3072)
        data = self.data_file.read(3072)

        return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order='F')