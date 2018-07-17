import numpy as np
from sklearn.metrics import roc_auc_score


def kl_divergence(y_true, y_pred):
    y_true = np.clip(y_true, np.finfo(float).eps, 1)
    y_pred = np.clip(y_pred, np.finfo(float).eps, 1)
    return np.sum(y_true * np.log(y_true / y_pred), axis=-1)


def dice_coefficient(x, y, smooth=1):
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])

    y = y > 0.5

    intersection = np.sum(x*y)
    return (2.*intersection+smooth) / (np.sum(x)+np.sum(y)+smooth)


def auroc_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return roc_auc_score(y_true, y_pred)