import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.signal import convolve2d


def k_fold_split(x_data, y_data, k=5):
    '''
    use this function
    usage : for i, (train_idx, val_idx) in enumerate(folds)
    '''
    assert x_data.shape[0] == y_data.shape[0]
    assert len(np.unique(y_data)) >= 2
    folds = list(StratifiedKFold(n_splits=k,
                                 shuffle=True,
                                 random_state=1).split(x_data, y_data))
    return folds


def split_dataset(x_data, y_data, size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=size, random_state=random_state)
    return [x_train, x_test, y_train, y_test]


def label_smoothing(y_data, epsilon=0.01):
    smooth_label = convolve2d(y_data, [[epsilon, 1.-2*epsilon, epsilon]], "same")
    return smooth_label
