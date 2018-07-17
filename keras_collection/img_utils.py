from skimage import io, exposure
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.misc import imresize
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def histogram_equalizer(img):
    return exposure.equalize_hist(img)


def load_img(path):
    return io.imread(path)


def save_img(img, path):
    return io.imsave(path, img)


def resize_img(img, img_h, img_w, method='bilinear'):
    return imresize(img, (img_h, img_w), method)


def minmax_img(img):
    scaler = MinMaxScaler()
    result = []
    for i in img:
        norm_img = scaler.fit_transform(i)
        result.append(norm_img)
    return np.array(result)


def mean_std_img(imgs, method='sample'):
    if method == 'sample':
        result = []
        for img in imgs:
            norm_img = (img-np.mean(img))/np.std(img)
            result.append(norm_img)
        result = np.array(result)
    else:
        result = (imgs-np.mean(imgs))/np.std(imgs)
    return result


def auroc_plot(y_true, y_pred, path="auroc_plot.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.step(fpr, tpr, "black", where="post",)

    plt.title("ROC Curve")
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(path)
    print("save fig")


