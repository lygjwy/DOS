'''
Downsampled variant (64 * 64) of ImageNet dataset (https://image-net.org/download-images)
Training: 1281167
Test: 50000
Num of classes: 1000
'''

import pickle
import numpy as np
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

def unpickle(data_file):
    with open(data_file, 'rb') as f:
        dict = pickle.load(f)
    return dict


class ImageNet64(Dataset):

    def __init__(self, root, transform=None, split='train'):

        data_dir = Path(root) / 'imagenet_64' / split

        self.transform = transform
        self.img_size = 64
        self.img_size2 = self.img_size * self.img_size
        self.labels = []
        self.imgs = []

        for idx in range(1, 11):
            data_file = data_dir / (split + '_data_batch_{}'.format(idx))
            img_data = unpickle(data_file)

            y = img_data['labels']
            y = [i-1 for i in y]

            x = img_data['data']
            x = np.dstack((x[:, :self.img_size2], x[:, self.img_size2:2 * self.img_size2], x[:, 2 * self.img_size2:]))
            x = x.reshape((x.shape[0], self.img_size, self.img_size, 3))
            self.labels.extend(y)
            self.imgs.append(x)
        
        self.imgs = np.concatenate(self.imgs, axis=0)
    
        self.num = len(self.labels)

    def __getitem__(self, index):

        img = self.imgs[index]
        img = Image.fromarray(img)

        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return {
            'data': img,
            'label': target
        }
    
    def __len__(self):
        return self.num