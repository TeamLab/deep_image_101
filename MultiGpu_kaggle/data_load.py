import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('labels.csv')
df_test = pd.read_csv('sample_submission.csv')

df_train.head(10)

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)

im_size = 299

x_train = []
y_train = []
x_test = []

i = 0

for f, breed in tqdm(df_train.values):
    img = cv2.imread('./train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.

num_class = y_train_raw.shape[1]

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw,
                                                      y_train_raw, test_size = 0.2,
                                                      random_state = 1)

savepath = os.getcwd()+"/data.npz"
np.savez(savepath,trainimg=X_train,trainlabel=Y_train,
        testimg=X_valid,testlabel=Y_valid,imgsize=im_size)

print("data saved to %s"%(savepath))