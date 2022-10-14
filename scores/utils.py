import torch

# OOD samples have smaller score
def get_kl_score(data_loader, clf):
    clf.eval()
    
    kl_score = []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            if args.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)
            
        softmax = torch.softmax(logit, dim=1)
        uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
        kl_score.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return kl_score

# OOD samples have larger weight
def get_msp_weight(data_loader, clf):
    clf.eval()

    msp_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            if clf.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)

        prob = torch.softmax(logit, dim=1)
        msp_weight.extend(torch.max(prob, dim=1)[0].tolist())

    return [1.0 - msp for msp in msp_weight]

# OOD samples have larger weight
def get_abs_weight(data_loader, clf):
    clf.eval()

    abs_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            if clf.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)

        prob = torch.softmax(logit, dim=1)
        abs_weight.extend(prob[:, -1].tolist())

    return abs_weight

# OOD samples have larger weight
def get_energy_weight(data_loader, clf):
    clf.eval()
    
    energy_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            if clf.include_binary:
                logit, _ = clf(data)
            else:
                logit = clf(data)
        energy_weight.extend((-torch.logsumexp(logit, dim=1)).tolist())
    
    return energy_weight

# OOD samples have larger weight
def get_binary_weight(data_loader, clf):
    clf.eval()

    binary_weight = []
    for sample in data_loader:
        data = sample['data'].cuda()

        with torch.no_grad():
            if clf.include_binary:
                _, energy_logit = clf(data)
            else:
                _ = clf(data)
        energy_prob = torch.sigmoid(energy_logit).tolist()
        binary_weight.extend(energy_prob)
    
    return binary_weight

weight_dic = {
    'msp': get_msp_weight,
    'abs': get_abs_weight,
    'energy': get_energy_weight,
    'binary': get_binary_weight
}

def get_weight(data_loader, clf, weight_type):
    if weight_type in weight_dic.keys():
        return weight_dic[weight_type](data_loader, clf)
    else:
        raise RuntimeError('<<< Invalid weight type {}'.format(weight_type))