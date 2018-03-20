'''
CIFAR-10 Image Category Dataset
The CIFAR-10 data ( https://www.cs.toronto.edu/~kriz/cifar.html ) contains 60,000 32x32 color images of 10 classes.
It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
Alex Krizhevsky maintains the page referenced here.
This is such a common dataset, that there are built in functions in TensorFlow to access this data.

Running this command requires an internet connection and a few minutes to download all the images.

'''
import os
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from skimage.feature import hog
from skimage import color

# import time

CF10_Labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# (CF10_X_train, CF10_y_train), (CF10_X_test, CF10_y_test) = tf.contrib.keras.datasets.cifar10.load_data()

# Next Example will show how many images in training (50,000), and show the first image (a frog)
# print(CF10_X_train.shape)
# print(CF10_y_train.shape)
# print(CF10_y_train[0,]) # this is a frog
# print(CF10_Labels[np.asscalar(CF10_y_train[0,])])

# Plot the 0-th image (a frog)
# from PIL import Image
# import matplotlib.pyplot as plt
# %matplotlib inline
# img = Image.fromarray(CF10_X_train[0,:,:,:])
# plt.imshow(img)

def feat_hog(ndImage):
    image = color.rgb2gray(ndImage)
    fd = hog(image, orientations=8, pixels_per_cell=(8, 8), visualise=False, cells_per_block=(1, 1), block_norm='L2-Hys')
    MinMaxScale = (fd-np.min(fd))/(np.max(fd)-np.min(fd))
    return MinMaxScale

def feat_color_hist(ndImage):
    np_hist,edges = np.histogramdd(ndImage.reshape(-1,3),bins=(3,3,3),normed=False,range=[(0,255),(0,255),(0,255)])
    MinMaxScale = (np_hist - np.min(np_hist))/(np.max(np_hist)-np.min(np_hist))
    return MinMaxScale.flatten() 


def train_test_dataset(feat_pair):
    savedfile = os.path.join(os.path.dirname(os.path.realpath(__file__)),'datasets','CIFAR10_pair_{}_{}.npz'.format(CF10_Labels[feat_pair[0]],CF10_Labels[feat_pair[1]]))
    if os.path.exists(savedfile):
        npzfile = np.load(savedfile)
        X_train, y_train, X_test, y_test = npzfile['X_train'], npzfile['y_train'], npzfile['X_test'], npzfile['y_test']
        npzfile.close()
    else:
        # Running this command requires an internet connection and a few minutes to download all the images.
        (CF10_X_train, CF10_y_train), (CF10_X_test, CF10_y_test) = tf.contrib.keras.datasets.cifar10.load_data()
        mask_train = (CF10_y_train[:,0] == feat_pair[0]) | (CF10_y_train[:,0] == feat_pair[1])
        y_train_raw = CF10_y_train[mask_train]
        X_train_raw = CF10_X_train[mask_train]
        mask_test = (CF10_y_test[:,0] == feat_pair[0]) | (CF10_y_test[:,0] == feat_pair[1])
        y_test_raw = CF10_y_test[mask_test]
        X_test_raw = CF10_X_test[mask_test]
        # 155 = hog((orient=8)* ((row_image/ppc)=32/8=4) * ((col_image/ppc)=32/8=4) = 128) + color_hist(bins channel1*2*3 = 3*3*3 = 27)
        X_train = np.zeros((X_train_raw.shape[0],155))
        X_test = np.zeros((X_test_raw.shape[0],155))
        mask_train_cls1 = (y_train_raw == feat_pair[0])
        y_train_raw[mask_train_cls1] = 1
        y_train_raw[~mask_train_cls1] = 0
        mask_test_cls1 = (y_test_raw == feat_pair[0])
        y_test_raw[mask_test_cls1] = 1
        y_test_raw[~mask_test_cls1] = 0
        for row in range(X_train_raw.shape[0]):
            f1 = feat_hog(X_train_raw[row,:,:,:])
            f2 = feat_color_hist(X_train_raw[row,:,:,:])
            f = np.concatenate((f1,f2),axis=0).reshape(1,-1)
            X_train[row] = f[0,:]
        for row in range(X_test_raw.shape[0]):
            f1 = feat_hog(X_test_raw[row,:,:,:])
            f2 = feat_color_hist(X_test_raw[row,:,:,:])
            f = np.concatenate((f1,f2),axis=0).reshape(1,-1)
            X_test[row] = f[0,:]
        y_train = y_train_raw
        y_test = y_test_raw
        save_pair_dataset_feat_extracted(feat_pair,X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test

def CF10_pairs(cls1, cls2):
    class_pair = []
    class_pair.append(CF10_Labels.index(cls1))
    class_pair.append(CF10_Labels.index(cls2))
    return class_pair

def save_pair_dataset_feat_extracted(feat_pair,X_train, y_train, X_test, y_test):
    datasets_path = 'datasets'
    if not(os.path.exists(datasets_path)):
        os.makedirs(datasets_path)
    file2save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'datasets','CIFAR10_pair_{}_{}.npz'.format(CF10_Labels[feat_pair[0]],CF10_Labels[feat_pair[1]]))
    if not(os.path.exists(file2save)):
        np.savez_compressed(file2save,X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# example implementation
# import AdaNet_CIFAR_10_feature_extraction as AdaFE
# import time
# cat_dog = AdaFE.CF10_pairs('cat','dog')

# start_time = time.time()
# X_train, y_train, X_test, y_test = AdaFE.train_test_dataset(cat_dog)
# print(time.time() - start_time)
# running time after images first load is 30sec

