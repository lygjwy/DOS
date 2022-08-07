import torch
import torch.nn.functional as F

def train_uni(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, beta=0.5):
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

def train_abs(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, beta=1.0):
    num_classes = len(data_loader_id.dataset.classes)
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)
        num_ood = sample_ood['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target_id = sample_id['label'].cuda()
        target_ood =  (torch.ones(num_ood) * num_classes).long().cuda()

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

        # evaluate
        _, pred = logit[:num_id, :num_classes].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target_id).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id.dataset), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id.dataset),
        'cla_acc': 100. * correct / total
    }

def train_trip(data_loader_id, data_loader_ood, net, optimizer, linear_optimizer, num_classes, beta=1.0, margin=1.0):
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

def get_trainer(name):
    if name == 'uni':
        return train_uni
    elif name == 'abs':
        return train_abs
    elif name == 'trip':
        return train_trip
    else:
        raise RuntimeError('<<< Invalid training method {}'.format(name))