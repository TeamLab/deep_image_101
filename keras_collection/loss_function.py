import numpy as np
from keras import backend as K
from keras import objectives
from scipy.spatial.distance import dice
from keras.models import *
from keras.layers import *


def segmetation_binary_loss(y_true, y_pred):
    y_true_flat = K.batch_flatten(y_true)
    y_pred_flat = K.batch_flatten(y_pred)
    return objectives.binary_crossentropy(y_true_flat, y_pred_flat)


def segmentation_dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = y_pred_f > 0.5
    y_pred_f = K.cast(y_pred_f, 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def segmentation_dice_loss(y_true, y_pred):
    return K.mean(1-segmentation_dice_coefficient(y_true, y_pred), axis=-1)


