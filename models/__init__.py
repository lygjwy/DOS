from .wide_resnet import Wide_ResNet, weights_init
from .wide_resnet_pretrain import WideResNet
from .densenet import DenseNet3

def get_clf(name, num_classes=10):
    if name == 'wrn40_2':
        # clf = Wide_ResNet(depth=40, widen_factor=2, dropout_rate=0.0, num_classes=num_classes)
        clf = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
    elif name == 'wrn40_4':
        clf = Wide_ResNet(depth=40, widen_factor=4, dropout_rate=0.0, num_classes=num_classes)
    elif name == 'densenet101':
        clf = DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0)
    else:
        raise RuntimeError('---> Invalid CLF name {}'.format(name))
    
    return clf