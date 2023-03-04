'''
80 million Tiny Images Dataset (https://groups.csail.mit.edu/vision/TinyImages/)
Code adapted from https://github.com/hendrycks/outlier-exposure
'''

import numpy as np
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

class TinyImages(Dataset):

    def __init__(self, root, transform=None, split='wo_cifar'):

        data_dir = Path(root) / 'tiny_images'
        
        if split == 'wo_cifar':
            data_file_path = data_dir / 'tiny_images_wo_cifar.bin'
            self.num = 79169608
        elif split == 'w_cifar':
            data_file_path = data_dir / 'tiny_images_w_cifar.bin'
            self.num = 132409
        else:
            data_file_path = data_dir / 'tiny_images.bin'
            self.num = 79302017

        self.data_file = open(data_file_path, 'rb')
        self.transform = transform

    def __getitem__(self, index):
        
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

        try:
            result = np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order='F')
        except ValueError:
            print(idx)
            exit()
        else:
            return result
