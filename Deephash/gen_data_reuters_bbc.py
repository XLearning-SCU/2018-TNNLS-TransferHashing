import os
import shutil
import sys
import random
from random import shuffle
from PIL import Image
# import tensorflow as tf
import io
import scipy.io as sio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# FUNCTION DEFINITION
def normalize_data(data):
    mean_vec = np.mean(data, axis=0)
    data_normalized = []
    for vec in data:
        v = vec - mean_vec
        if np.linalg.norm(v) == 0:
            norm1 = v
        else:
            norm1 = v / np.linalg.norm(v)
        data_normalized.append(norm1)
    return data_normalized

def my_shuffle(x):
    idx = list(range(len(x)))
    shuffle(idx)
    x_tmp = []
    for i in idx:
        x_tmp.append(x[i])
    return x_tmp


def my_loadmat(fname, var_name):
    return sio.loadmat(fname)[var_name]

# SELECT DATASET
# dataset = 'bbc'
dataset = 'reuters'

if dataset == 'reuters':
    NORMALIZE_DATA = True   # NORMALIZE DATA FOR REUTERS DATASET
    fn_data_all_raw_img = 'db_data_img.mat'
    fn_data_all_raw_txt = 'db_data_tag.mat'
    img_var_name = 'db_data_img'
    txt_var_name = 'db_data_tag'
    data_all_raw_img = sio.loadmat(fn_data_all_raw_img)[img_var_name]
    data_all_raw_txt = sio.loadmat(fn_data_all_raw_txt)[txt_var_name]
    NUM_ALL = len(data_all_raw_img)
    NUM_TRAIN = 10000
    NUM_TEST = 1000
elif dataset == 'bbc':
    NORMALIZE_DATA = False  # DO NOT NORMALIZE DATA FOR BBC DATASET
    fn_data_all_raw_img = 'bbc_PCA_6_2of3.mat'
    fn_data_all_raw_txt = 'bbc_PCA_6_1of3.mat'
    img_var_name = 'PCA_feature'
    txt_var_name = 'PCA_feature'
    data_all_raw_img = sio.loadmat(fn_data_all_raw_img)[img_var_name]
    data_all_raw_txt = sio.loadmat(fn_data_all_raw_txt)[txt_var_name]
    NUM_ALL = len(data_all_raw_img) if len(data_all_raw_img) == len(data_all_raw_txt) else min(len(data_all_raw_img), len(data_all_raw_txt))
    NUM_TRAIN = 1000
    NUM_TEST = NUM_ALL - NUM_TRAIN
else:
    raise('invalid dataset name')

print(np.shape(data_all_raw_img))
print(np.shape(data_all_raw_txt))

# NORMALIZE DATA
if NORMALIZE_DATA:
    data_all_raw_img = normalize_data(data_all_raw_img)
    data_all_raw_txt = normalize_data(data_all_raw_txt)

data_all_raw = []
for i in range(NUM_ALL):
    data_all_raw.append([data_all_raw_img[i], data_all_raw_txt[i]])

print(np.shape(data_all_raw))

data_all_raw = my_shuffle(data_all_raw)

data_train_chosen = data_all_raw[:NUM_TRAIN]
data_test_chosen = data_all_raw[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
np.save('data_train_chosen_' + dataset + '.npy',data_train_chosen)
np.save('data_test_chosen_' + dataset + '.npy',data_test_chosen)

print(np.shape(data_train_chosen))
print(np.shape(data_test_chosen))

