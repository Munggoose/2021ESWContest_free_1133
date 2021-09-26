""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

def evaluate(labels, scores, RGB_score, epoch4Test, ab_thres, is_abList, use_abscore, opt, metric='RGB'):
    if metric == 'roc':
        return roc(labels, scores, epoch4Test)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    elif metric == 'RGB':
        if (epoch4Test == 1 or epoch4Test % opt.save_evalRGB_freq == 0):
            evaluate_RGB(labels, RGB_score, epoch4Test, ab_thres, is_abList, use_abscore)
        return roc(labels, scores, epoch4Test)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def evaluate_RGB(labels, scores, epoch4Test, ab_thres, is_abList, use_abscore):
    """[summary] saves ROC-curve as (epoch).png

    Args:
        labels ([type]): [description]
        scores ([type]): [description]
        epoch4Test ([type]): [description]
        ab_thres ([type]): [description]
    """    
    plt.clf()
    labels = labels.cpu()
    mapped = [0 for i in range(len(scores))]
    tnr = []
    tpr = []
    fpr = []
    x = []
    
    Max_f1 = 0
    prop_thres = 0
    term = 100000
    thres_range = 800 #0.00001~0.01000

    # for thres in range(20000, 30000, 1):
    for thres in range(thres_range):
        # if thres_range % thres/10 == 0:
        # print(f'{thres / thres_range * 100: .2f}%')
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        
        for i in range(len(scores)):
            if ((use_abscore and scores[i] >= thres / term) or ((not use_abscore) and is_abList[i])): #abnormal
                if(labels[i] == 0):
                    tp += 1
                else:
                    fp += 1
            else:
                if(labels[i] == 1):
                    tn += 1
                else:
                    fn += 1
        
        tnr.append(fp / (tn+fp + 1e-4))
        tpr.append(tp / (tp+fn + 1e-4))
        fpr.append(1 - (fp / (fp+tn + 1e-4)))
        x.append(thres / (term + 1e-4))

        curr_f1 = tp / (tp+(fp+fn)/2 + 1e-4)
        if (curr_f1 > Max_f1):
            Max_f1 = curr_f1
            prop_thres = thres / term
        if(thres/term == ab_thres):
            print(f'thres: {thres/term}')
            print(f'tn: {tn}\tfp: {fp}')
            print(f'fn: {fn}\ttp: {tp}')
            print(f'f1-score: {tp / (tp+(fp+fn)/2 + 1e-4)}')
            # print(new_f1)
    plt.scatter(tnr, tpr, c='red')
    plt.savefig('./output/ganomaly/casting/test/roc/' + str(epoch4Test) + '.png')
    
    print(f'Max_f1: {Max_f1}')
    print(f'prop_thres: {prop_thres}')
    plt.clf()
    plt.scatter(x, tpr, c='red')
    plt.scatter(x, fpr, c='blue')
    plt.xlabel('Threshold')
    plt.ylabel('True Positive Rate')
    # plt.show()

def roc(labels, scores, epoch4Test, saveto='./',):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.clf()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        
        if (epoch4Test == 1 or epoch4Test % 20 == 0):
            plt.savefig('./output/ganomaly/casting/test/roc/abscore/' + str(epoch4Test) + '.png')
        plt.close()

    return roc_auc

    # labels = labels.cpu()
    # print(roc_curve(labels, scores))

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap
