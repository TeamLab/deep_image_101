import os
import numpy as np
from scipy.misc import imread , imresize


workspace = os.getcwd()
path_image = workspace + '/data/image/'
path_annotation = workspace+'/data/annotation/'
file_image_list = os.listdir(path_image)
file_annotation_list = os.listdir(path_annotation)
file_image_list.sort()
file_annotation_list.sort()


def read_image():
    n_image = 0
    for f in file_image_list:
        fullpath =path_image+f
        image = imread(fullpath)
        image = imresize(image,[224,224])
        image_vector = np.reshape(image, (1, -1))
        if n_image is 0:
            total_image = image_vector
        else:
            total_image = np.concatenate((total_image,image_vector),axis=0)
        n_image = n_image+1
    print('Total image number is %d'%n_image)
    train_image = total_image[:int(0.8 * len(total_image)), :]
    test_image = total_image[int(0.8 * len(total_image)):, :]
    print('Train image number :', len(train_image))
    print('Test image number :', len(test_image))
    return train_image, test_image


def read_annotation():
    n_annotation = 0
    for f in file_annotation_list:
        fullpath =path_annotation+f
        annotation = imread(fullpath)
        annotation = imresize(annotation,[224,224])
        annotation_vector = np.reshape(annotation, (1, -1))
        if n_annotation is 0:
            total_annotation = annotation_vector
        else:
            total_annotation = np.concatenate((total_annotation,annotation_vector),axis=0)
        n_annotation = n_annotation+1
    print('Total annotation number is %d'%n_annotation)
    train_annotation = total_annotation[:int(0.8 * len(total_annotation)), :]
    test_annotation = total_annotation[int(0.8 * len(total_annotation)):, :]
    print('Train annotation number : ', len(train_annotation))
    print('Test annotation number : ', len(test_annotation))
    return train_annotation,test_annotation


