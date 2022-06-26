from .wide_resnet import Wide_ResNet

def get_clf(name, num_classes):
    if name == 'wrn40':
        clf = Wide_ResNet(depth=40, widen_factor=4, dropout_rate=0.3, num_classes=num_classes)
    else:
        raise RuntimeError('---> Invalid CLF name {}'.format(name))
    
    return clf