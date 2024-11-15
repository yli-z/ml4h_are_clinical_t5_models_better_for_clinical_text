"""
Metric Function for HoC is from:
https://github.com/michiyasunaga/LinkBERT/blob/main/src/seqcls/utils_hoc.py
"""

import numpy as np


classes = ['evading growth suppressors', 
            'tumor promoting inflammation', 
            'enabling replicative immortality', 
            'cellular energetics', 
            'resisting cell death', 
            'activating invasion and metastasis', 
            'genomic instability and mutation', 
            '', 
            'inducing angiogenesis', 
            'sustaining proliferative signaling', 
            'avoiding immune destruction']

def divide(x, y):
    return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float64), where=y != 0)

def compute_p_r_f(preds, labels):
    TP = ((preds == labels) & (preds != 0)).astype(int).sum()
    P_total = (preds != 0).astype(int).sum()
    L_total = (labels != 0).astype(int).sum()

    P  = divide(TP, P_total).mean()
    R  = divide(TP, L_total).mean()
    F1 = divide(2 * P * R, (P + R)).mean()

    return P, R, F1

def hoc_metric(targets, predictions):  
    data = {}
    label_range = [x for x in range(11)]

    for i, (p,t) in enumerate(zip(predictions, targets)):

        p = [classes.index(x) for x in p.split(', ')]
        t = [classes.index(x) for x in t.split(', ')]
        p = set(p)
        t = set(t)
        data[i] = (t,p)
    
    y_test = []
    y_pred = []
    for k, (true, pred) in data.items():
        t = [0] * len(label_range)
        for i in true:
            if  i >= len(label_range):
                continue
            t[i] = 1

        p = [0] * len(label_range)
        for i in pred:
            if i >= len(label_range):
                continue
            p[i] = 1

        y_test.append(t)
        y_pred.append(p)
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    p, r, f1 = compute_p_r_f(y_pred, y_test)
    return {"precision": p, "recall": r, "F1": f1}