from torchvision import transforms

from .tiny_images import TinyImages
from .named_dataset_with_meta import NamedDatasetWithMeta

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def get_ds_info(ds_name, info_type):
    ds_infos = {
        'svhn': {
            'image_size': 224,
            'channel': 3,
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'mean_and_std': [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)]
        },
        'cifar10': {
            'image_size': 32,
            'channel': 3,
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'mean_and_std': [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]
        },
        'cifar100': {
            'image_size': 32,
            'channel': 3,
            'classes': CIFAR100_CLASSES,
            'mean_and_std': [(0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)]
        }
    }

    if ds_name not in ds_infos.keys():
        raise Exception('---> Dataset {} info not avaliable.'.format(ds_name))

    ds_info = ds_infos[ds_name]
    if info_type not in ds_info.keys():
        raise Exception('---> Dataset {} info type {} not available.'.format(ds_name, info_type))
    
    return ds_info[info_type]

def get_ds_trf(ds_name, stage):
    mean, std = get_ds_info(ds_name, 'mean_and_std')
    if stage == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif stage == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise RuntimeError('---> Invalid stage {}'.format(stage))

def get_ds(root, ds_name, split, transform, target_transform=None):
    
    if ds_name == 'tiny_images':
        print('---> Loading Tiny Images (split={} for partition)'.format(split))
        ds = TinyImages(root, transform, split=split)
    else:
        ds = NamedDatasetWithMeta(
            root=root,
            name=ds_name,
            split=split,
            transform=transform,
            target_transform=target_transform
        )

    return ds


def get_ood_trf(ds_name_id, ds_name_ood, stage):
    mean, std = get_ds_info(ds_name_id, 'mean_and_std')

    if stage == 'train':
        ood_trf = {
            'tiny_images': [
                # transforms.ToTensor(), 
                # transforms.ToPILImage(), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomCrop(32, padding=4), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ],
        }
    elif stage == 'test':
        ood_trf = {
            'svhn': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'places365_10k': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'lsunc': [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'lsunr': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'tinc': [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'tinr': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'dtd': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'isun': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'cifar10': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'cifar100': [transforms.ToTensor(), transforms.Normalize(mean, std)]
        }
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))

    return transforms.Compose(ood_trf[ds_name_ood])
