import time
import numpy as np

import torch
import torch.nn.functional as F

def train_uni(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=0.5):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target)
        loss += beta * -(logit[num_id:].mean(dim=1) - torch.logsumexp(logit[num_id:], dim=1)).mean()

        # backward
        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if linear_scheduler is not None:
            linear_scheduler.step()

        # evaluate
        _, pred = logit[:num_id].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def train_abs(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=1.0):
    num_classes = len(data_loader_id.dataset.classes)
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)
        num_ood = sample_ood['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target_id = sample_id['label'].cuda()
        target_ood = (torch.ones(num_ood) * num_classes).long().cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target_id)
        loss += beta * F.cross_entropy(logit[num_id:], target_ood)

        # backward
        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if linear_scheduler is not None:
            linear_scheduler.step()

        # evaluate
        _, pred = logit[:num_id, :num_classes].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target_id).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def train_energy(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=0.1):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda()

        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target)
        Ec_in = -torch.logsumexp(logit[:num_id], dim=1)
        Ec_out = -torch.logsumexp(logit[num_id:], dim=1)
        m_in = -25
        m_out = -7
        loss += beta * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())

        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if linear_scheduler is not None:
            linear_scheduler.step()
        
        _, pred = logit[:num_id].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def train_trip(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=0.5, margin=10.0):
    num_classes = len(data_loader_id.dataset.classes)
    net.train()

    total, correct = 0, 0
    total_cla_loss, total_trip_loss = 0.0, 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0) # [BATCH_ID,]

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda() # [BATCH_ID + BATCH_OOD, PIX]
        target = sample_id['label'].cuda() # [BATCH_ID,]

        # forward
        logit = net(data) # [BATCH_ID + BATCH_OOD, NUM_CLA]
        
        cla_loss = F.cross_entropy(logit[:num_id], target) # classification
        # calculate regularization term loss
        trip_loss_list = torch.empty([0]).cuda()
        for i in range(num_classes):
            valid_mask = (target == i)
            if torch.count_nonzero(valid_mask) > 0:
                # logits = - euclidean distance
                ap_logits = logit[:num_id][valid_mask, i].unsqueeze(1) # [BATCH_ID_K, 1]
                an_logits = logit[num_id:][:, i] # [BATCH_OOD, ]
                
                # penul_feat_ood = penul_feat[num_id:] # [BATCH_OOD, FEAT_DIM]
                # w_i = net.linear.weight.T[:, i] # [FEAT_DIM,]
                # an_logits = -((penul_feat_ood - w_i).pow(2)).mean(dim=1) #  [BATCH_OOD,]

                pa_logits = (ap_logits - an_logits - margin).view(-1, 1) # [BATCH_ID_K * BATCH_OOD, 1]
                a_target = torch.ones_like(pa_logits).cuda()
                trip_loss_list = torch.cat((trip_loss_list, F.binary_cross_entropy_with_logits(pa_logits, a_target).unsqueeze(0)), dim=0)
        # print(torch.mean(trip_loss_list))
        trip_loss = beta * torch.mean(trip_loss_list)
        loss = cla_loss + trip_loss

        # backward
        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if linear_scheduler is not None:
            linear_scheduler.step()
        
        # evaluate
        _, pred = logit[:num_id,].max(dim=1)
        with torch.no_grad():
            total_cla_loss += cla_loss.item()
            total_trip_loss += trip_loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id
    
    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}% | trip loss: {:.8f}]'.format(total_cla_loss / len(data_loader_id), 100. * correct / total, total_trip_loss / len(data_loader_id)))
    return {
        'cla_loss': total_cla_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def train_contra(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=1.0):
    # num_classes = len(data_loader_id.dataset.classes)
    net.train()

    total, correct = 0, 0
    total_cla_loss, total_contra_loss = 0.0, 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda() # [NUM_ID,]

        # forward
        logit = net(data)
        cla_loss = F.cross_entropy(logit[:num_id], target)
        # regularization term loss
        cat_logits = torch.zeros([0])
        for i in range(num_id):
            cat_idx = target[i]

            cat_id_logit = logit[i, cat_idx].view(1, -1)
            cat_ood_logits = logit[num_id:, cat_idx].view(1, -1)
            cat_logit = torch.cat([cat_id_logit, cat_ood_logits], dim=1) # [1, 1+NUM_OOD]

            if i == 0:
                cat_logits = cat_logit
            else:
                cat_logits = torch.cat([cat_logits, cat_logit], dim=0)
        
        # cat_logits: [NUM_ID, 1+NUM_OOD]
        cat_targets = torch.zeros((num_id,), dtype=torch.long).cuda()
        contra_loss = F.cross_entropy(cat_logits, cat_targets)
        loss = cla_loss + beta * contra_loss

        # backward
        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if linear_scheduler is not None:
            linear_optimizer.step()

        # evaluate
        _, pred = logit[:num_id,].max(dim=1)
        with torch.no_grad():
            total_cla_loss += cla_loss.item()
            total_contra_loss += contra_loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}% | contra loss: {:.8f}]'.format(total_cla_loss / len(data_loader_id), 100. * correct / total, total_contra_loss / len(data_loader_id)))
    return {
        'cla_loss': total_cla_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def centroid_estimator(data_loader, clf):
    num_classes = len(data_loader.dataset.classes)
    clf.eval()

    num_sample_per_class = np.zeros(num_classes)
    list_feats = [0] * num_classes

    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()

        with torch.no_grad():
            _, feat = clf(data, ret_feat=True)

        for i in range(target.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                list_feats[label] = feat[i].view(1, -1)
            else:
                list_feats[label] = torch.cat((list_feats[label], feat[i].view(1, -1)), dim=0)
            num_sample_per_class[label] += 1
        
    category_sample_mean = []
    for j in range(num_classes):
        category_sample_mean.append(torch.mean(list_feats[j], 0).tolist())
    
    return torch.tensor(category_sample_mean)

def train_euc(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=1.0):
    # num_classes = len(data_loader_id.dataset.classes)
    temp = 10.0
    net.train()

    # total, correct = 0, 0
    total_conc_loss, total_disp_loss = 0.0, 0.0

    # estimate class centrois each iteration
    # t0 = time.time()
    cat_sample_mean = centroid_estimator(data_loader_id, net) # [NUM_CLASSES, FEAT_DIM]
    # print(time.time() - t0)

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):  

        num_id = sample_id['data'].size(0)
        data_id = sample_id['data'].cuda() # [NUM_ID, C, H, W]
        target = sample_id['label'].cuda() # [NUM_ID,]

        data_ood = sample_ood['data'].cuda() # [NUM_OOD, C, H, W]
        _, feat_id = net(data_id, ret_feat=True) # [NUM_ID, FEAT_DIM]
        _, feat_ood = net(data_ood, ret_feat=True) # [NUM_OOD, FEAT_DIM]

        cat_sample_mean_ = cat_sample_mean.T.unsqueeze(0).cuda() # [1, FEAT_DIM, NUM_CLA]
        feat_id_ = feat_id.unsqueeze(2) # [NUM_ID, FEAT_DIM, 1]
        feat_ood_ = feat_ood.unsqueeze(2) # [NUM_OOD, FEAT_DIM, 1]
        logit_id = -((feat_id_ - cat_sample_mean_).pow(2)).mean(1) # [NUM_ID, NUM_CLASSES]
        logit_ood = -((feat_ood_ - cat_sample_mean_).pow(2)).mean(1) # [NUM_OOD, NUM_CLASSES]
        
        # concentration loss
        conc_loss = F.cross_entropy(logit_id / temp, target)

        # dispersion loss
        cat_logits = torch.zeros([0])
        for i in range(num_id):
            cat_idx = target[i]

            cat_id_logit = logit_id[i, cat_idx].view(1, -1)
            cat_ood_logits = logit_ood[:, cat_idx].view(1, -1)
            cat_logit = torch.cat([cat_id_logit, cat_ood_logits], dim=1)

            if i == 0:
                cat_logits = cat_logit
            else:
                cat_logits = torch.cat([cat_logits, cat_logit], dim=0)
        
        cat_targets = torch.zeros((num_id,), dtype=torch.long).cuda()
        disp_loss = F.cross_entropy(cat_logits / temp, cat_targets)
        
        # combine the concentration & dispersion loss
        loss = conc_loss + beta * disp_loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # evaluate
        with torch.no_grad():
            print(conc_loss.item(), disp_loss.item())
            total_conc_loss += conc_loss.item()
            total_disp_loss += disp_loss.item()
        
    print('[concentration loss: {:.8f} | dispersion loss: {:.8f}]'.format(total_conc_loss / len(data_loader_id), total_disp_loss / len(data_loader_id)))

def train_binary(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, scheduler=None, linear_scheduler=None, beta=1.0):
    net.train()

    total, correct = 0, 0
    total_cla_loss, total_binary_loss = 0.0, 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)
        num_ood = sample_ood['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda()

        # forward
        logit, _, energy_logit = net(data, ret_feat=True, ret_el=True) # [NUM, 1]
        cla_loss = F.cross_entropy(logit[:num_id], target)
        # binary classification loss
        # binary_target = torch.cat([torch.zeros([num_id,], dtype=torch.long), torch.ones([num_ood,], dtype=torch.long)], dim=0).cuda()
        # binary_loss = F.cross_entropy(energy_logit, binary_target)

        binary_target = torch.cat([torch.zeros([num_id,]), torch.ones([num_ood,])], dim=0).unsqueeze(1).cuda()
        binary_loss = F.binary_cross_entropy_with_logits(energy_logit, binary_target)
        
        loss = cla_loss + beta * binary_loss

        # backward
        optimizer.zero_grad()
        linear_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        linear_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if linear_scheduler is not None:
            linear_optimizer.step()

        # evaluate
        _, pred = logit[:num_id,].max(dim=1)
        with torch.no_grad():
            total_cla_loss += cla_loss.item()
            total_binary_loss += binary_loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}% | binary loss: {:.8f}]'.format(total_cla_loss / len(data_loader_id), 100. * correct / total, total_binary_loss / len(data_loader_id)))
    return {
        'cla_loss': total_cla_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def get_trainer(name):
    if name == 'uni':
        return train_uni
    elif name == 'abs':
        return train_abs
    elif name == 'energy':
        return train_energy
    elif name == 'trip':
        return train_trip
    elif name == 'contra':
        return train_contra
    elif name == 'euc':
        return train_euc
    elif name == 'binary':
        return train_binary
    else:
        raise RuntimeError('<<< Invalid training method {}'.format(name))