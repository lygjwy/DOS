from .wide_resnet import Wide_ResNet, Wide_ResNet_Binary, weights_init

def get_clf(name, num_classes, include_binary=False):
    if name == 'wrn40':
        if include_binary:
            clf = Wide_ResNet_Binary(depth=40, widen_factor=4, dropout_rate=0.0, num_classes=num_classes)
        else:
            clf = Wide_ResNet(depth=40, widen_factor=4, dropout_rate=0.0, num_classes=num_classes)
    else:
        raise RuntimeError('---> Invalid CLF name {}'.format(name))
    
    return clf