import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

df = pd.read_csv('labels.csv')
df.head()

n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

width = 32
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
for i in tqdm(range(n)):
    X[i] = cv2.resize(cv2.imread('train/%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1

X = np.array(X, np.float32) / 255.
savepath = os.getcwd()+"/data.npz"
np.savez(savepath, trainimg=X, trainlabel=y)

print("data saved to %s" % savepath)

