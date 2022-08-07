from .wide_resnet import Wide_ResNet

def get_clf(name, num_classes, clf_type='inner'):
    if name == 'wrn40':
        clf = Wide_ResNet(depth=40, widen_factor=4, dropout_rate=0.0, num_classes=num_classes, clf_type=clf_type)
    else:
        raise RuntimeError('---> Invalid CLF name {}'.format(name))
    
    return clf