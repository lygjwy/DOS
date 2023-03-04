import numpy as np
from pathlib import Path
from PIL import Image

data_dir = Path('/data/cv/tiny_images')

split = 'wo_cifar'
data_file_path = data_dir / ('tiny_images_' + split + '.bin')

np.random.seed(14)
if split == 'wo_cifar':
    idxs = np.random.randint(low=0, high=79169608, size=100)
elif split == 'w_cifar':
    idxs = np.random.randint(low=0, high=132409, size=100)

def load_img(data_file, idx):
    data_file.seek(idx * 3072)
    data = data_file.read(3072)

    result = np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order='F')
    return result

with open(data_file_path, 'rb') as data_file:
    for idx in idxs:
        img_path = Path('./'+split) / (str(idx) + '.png')
        img = load_img(data_file, idx)
        img = Image.fromarray(img)

        img.save(img_path)



