import ast
import os
from PIL import Image
from pathlib import Path

from torchvision.datasets import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_img_file(file_path):
    # check if the image path valid (file exists and has allowed extension)
    if os.path.exists(file_path) and has_file_allowed_extension(file_path, IMG_EXTENSIONS):
        return True
    return False


class NamedDatasetWithMeta(VisionDataset):
    """A custom dataset where the samples are arranged in this way:
    labeled:
    root/folder_name/split/category/1.ext
    
    unlabeled:
    root/folder_name/split/1.ext

    meta:
    classes.txt for labeled
    train.txt records path for train split samples
    test.txt records path for test split samples

    Attributes:
        classes (list): sorted in alphabetical order
        class_to_idx (dic):
        samples (tuple list): [(image_path, target) | (image_path) or other formats]
    """
    def __init__(self, root, name, split, transform, target_transform=None):
        root = os.path.expanduser(root)
        super(NamedDatasetWithMeta, self).__init__(root, transform=transform, target_transform=target_transform)
        self.name = name
        
        #  load classes info
        self.root = Path(root)
        dataset_path = self.root / self.name
        
        # load dataset samples & targets info
        if split == 'train':
            self.entry_path = dataset_path / 'train.txt'
            self.complexity_path = dataset_path / 'train_complexity.txt'
        elif split == 'test':
            self.entry_path = dataset_path / 'test.txt'
            self.complexity_path = dataset_path / 'test_complexity.txt'
        else:
            raise RuntimeError('<--- Invalid split: {}'.format(split))

        if not self.entry_path.is_file():
            raise RuntimeError('<--- Split entry file: {} not exist.'.format(str(self.entry_path)))
        
        self.classes, self.class_to_idx = self._find_classes(dataset_path)
        
        # read entry data from entry file
        self.samples = self._parse_entry_file()


    def _find_classes(self, dataset_path):
        classes_path = dataset_path / 'classes.txt'

        if classes_path.is_file():
            # read images classes info
            self.labeled = True
            with open(classes_path, 'r') as f:
                classes = sorted(ast.literal_eval(f.readline()))
            class_to_idx = {cla: idx for idx, cla in enumerate(classes)}
        else:
            self.labeled = False
            classes = None
            class_to_idx = None

        return classes, class_to_idx


    def _parse_entry_file(self):
        
        with open(self.entry_path, 'r') as ef:
            entries = ef.readlines()
        
        tokens_list = [entry.strip().split(' ') for entry in entries]
        samples = []

        for tokens in tokens_list:
            img_path = str(self.root / tokens[0].replace('\\', '/'))
            
            if is_img_file(img_path):
                if self.labeled:
                    samples.append((img_path, int(tokens[1])))
                else:
                    samples.append((img_path,))
            else:
                raise RuntimeError('<--- invalid image path: {}'.format(img_path))
    
        return samples
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        image_path = sample[0]
        image = default_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        
        if self.labeled:
            target = sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return {'data': image, 'label': target}
        else:
            return {'data': image}
    
    def __len__(self):
        return len(self.samples)
