'''
Metrics for out-of-distribution detection
'''

import numpy as np
import sklearn.metrics

def fpr_tpr(score, label, tpr):
    score_id = score[label == 1]
    score_ood = score[label == 0]
    len_id = len(score_id)
    len_ood = len(score_ood)

    num_tp = int(np.floor(tpr * len_id))
    th = np.sort(score_id)[-num_tp]

    num_fp = np.sum(score_ood > th)
    fpr = num_fp / len_ood

    return fpr

def auc_roc_pr(score, label):
    # indicator_id = np.zeros_like(label)
    # indicator_id

    fpr, tpr, _ = sklearn.metrics.roc_curve(label, score)
    prec_id, rec_id, _ = sklearn.metrics.precision_recall_curve(label, score)
    prec_ood, rec_ood, _ = sklearn.metrics.precision_recall_curve(1 - label, -score)

    auroc = sklearn.metrics.auc(fpr, tpr)
    aupr_id = sklearn.metrics.auc(rec_id, prec_id)
    aupr_ood = sklearn.metrics.auc(rec_ood, prec_ood)

    return auroc, aupr_id, aupr_ood

def compute_all_metrics(score, label, verbose=True):
    tpr = 0.95
    fpr_at_tpr = fpr_tpr(score, label, tpr)
    auroc, aupr_id, aupr_ood = auc_roc_pr(score, label)

    if verbose:
        print('[auroc: {:.4f}, aupr_in: {:.4f}, aupr_out: {:.4f}, fpr@95tpr: {:.4f}]'.format(auroc, aupr_in, aupr_out, fpr_at_tpr))
        results = [fpr_at_tpr, auroc, aupr_id, aupr_ood]
    
    return results
