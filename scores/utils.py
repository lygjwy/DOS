import torch
import torch.nn.functional as F

# OOD samples have smaller score
def get_kl_score(data_loader, clf):
    clf.eval()
    
    kl_score = []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            
            logit = clf(data)
            
        softmax = torch.softmax(logit, dim=1)
        uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
        kl_score.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return kl_score

# OOD samples have larger weight
def get_msp_weight(data_loader, clf, ret_feat):
    clf.eval()

    msp_weight = []
    feats = []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            
            if ret_feat:
                logit, out = clf(data, ret_feat)
            else:
                logit = clf(data, ret_feat)

        prob = torch.softmax(logit, dim=1)
        msp_weight.extend(torch.max(prob, dim=1)[0].tolist())

        if len(feats) == 0:
            feats = out
        else:
            feats = torch.cat([feats, out], dim=0)
    
    if ret_feat:
        return [1.0 - msp for msp in msp_weight], feats
    else:
        return [1.0 - msp for msp in msp_weight]

# OOD samples have larger weight
def get_abs_weight(data_loader, clf, ret_feat):
    clf.eval()

    abs_weight = []
    feats = []

    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            
            if ret_feat:
                logit, out = clf(data, ret_feat)
            else:
                logit = clf(data, ret_feat)

        prob = torch.softmax(logit, dim=1)
        abs_weight.extend(prob[:, -1].tolist())

        if len(feats) == 0:
            feats = out
        else:
            feats = torch.cat([feats, out], dim=0)
    
    if ret_feat:
        return abs_weight, feats
    else:
        return abs_weight

# OOD samples have larger weight
def get_energy_weight(data_loader, clf, ret_feat):
    clf.eval()
    
    energy_weight = []
    feats = []
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            
            if ret_feat:
                logit, out = clf(data, ret_feat)
            else:
                logit = clf(data, ret_feat)
        
        energy_weight.extend((-torch.logsumexp(logit, dim=1)).tolist())
    
        if len(feats) == 0:
            feats = out
        else:
            feats = torch.cat([feats, out], dim=0)

    if ret_feat:
        return energy_weight, feats
    else:
        return energy_weight

weight_dic = {
    'msp': get_msp_weight,
    'abs': get_abs_weight,
    'energy': get_energy_weight
}

def get_weight(data_loader, clf, weight_type, ret_feat=False):
    if weight_type in weight_dic.keys():
        return weight_dic[weight_type](data_loader, clf, ret_feat)
    else:
        raise RuntimeError('<<< Invalid weight type {}'.format(weight_type))