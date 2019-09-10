
from __future__ import print_function
import os
import shutil
import sys
from random import shuffle
from PIL import Image
import tensorflow as tf
import io
import scipy.io as sio
import numpy as np
from numpy import linalg as LA
#import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name', default=None, required=True)
parser.add_argument('--ckpt', help='checkpoint epoch',  default=-1)
args = parser.parse_args()

datasets = ['bbc','reuters']
dataset = args.dataset
ckpt = args.ckpt

if dataset not in datasets:
    raise ('dataset invalid!')

print('feeding for %s\ncheckpoint: %s'%(dataset,ckpt))

saved_model_path = './save_%s'%dataset
meta_name = 'mlp-%s.meta'%ckpt

#make sure mat name and matlab varible name are same
def measure_identity(w):
    n = np.shape(w)[1]
    mat = 1./n * np.matmul(np.transpose(w), w)
    err = 1./n**2 * LA.norm(mat - np.eye(n))**2
    return err, mat

USE_TRAINING_SET = True

if USE_TRAINING_SET:
    train_data_raw = np.load('data_train_chosen_%s.npy'%dataset)
    test_data_raw = np.load('data_test_chosen_%s.npy'%dataset)

    train_data = []
    test_data = []
    for item in train_data_raw:
        train_data.append(item[0])
    for item in test_data_raw:
        test_data.append(item[0])
    train_data_raw = []
    test_data_raw = []
    print('diff btwn train 0/1: ', sum(train_data[0]-test_data[1]))
    print('diff btwn test  0/n: ', sum(train_data[0]-test_data[100]))
    print('len train:', len(train_data))
    print('len test: ',len(test_data))
else:
    pass
    # CHOOSE OTHER DATA TO HASH
    pth = './cm_nus_train_test_rand.mat'
    train_data = sio.loadmat(pth)['train_rand_batch']
    test_data = sio.loadmat(pth)['test_rand_batch']

train_data = train_data[:30000]
test_data = test_data[:30000]

sess=tf.Session()
graph = tf.get_default_graph()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(os.path.join(saved_model_path, meta_name))
saver.restore(sess,tf.train.latest_checkpoint(saved_model_path))

# va = graph.get_collection(tf.GraphKeys.VARIABLES)
input_x = graph.get_tensor_by_name("input_x:0")
# for item in graph.get_operations():
#     print(item)

b = graph.get_tensor_by_name("mlp/b:0")

fd_train ={input_x:train_data}
fd_test ={input_x:test_data}

bs_train = sess.run(b,feed_dict=fd_train)
bs_test = sess.run(b,feed_dict=fd_test)

h1 = sess.run(graph.get_tensor_by_name("mlp/h1:0"))
h2 = sess.run(graph.get_tensor_by_name("mlp/h2:0"))
h3 = sess.run(graph.get_tensor_by_name("mlp/out:0"))

b1 = sess.run(graph.get_tensor_by_name("mlp/b1:0"))
b2 = sess.run(graph.get_tensor_by_name("mlp/b2:0"))
b3 = sess.run(graph.get_tensor_by_name("mlp/b_out:0"))

print('h1', np.shape(h1))
print('h2', np.shape(h2))
print('h3', np.shape(h3))

print('b1', np.shape(b1))
print('b2', np.shape(b2))
print('b3', np.shape(b3))

#print(h3[0])
print(len(h3[0]))
print('b3',b3)

print('dot sim btwn h3 rows 0/1: ', sum(i[0] * i[1] for i in zip(h3[0], h3[1])))
print('dot sim btwn h3 rows 0/30: ', sum(i[0] * i[1] for i in zip(h3[0], h3[30])))
print('dot sim btwn h3 rows 0/50: ', sum(i[0] * i[1] for i in zip(h3[0], h3[50])))
print('dot sim btwn h3 rows 30/50: ', sum(i[0] * i[1] for i in zip(h3[30], h3[50])))

print('dot sim btwn h3 rows 0/0: ', sum(i[0] * i[1] for i in zip(h3[0], h3[0])))
print('dot sim btwn h3 rows 30/30: ', sum(i[0] * i[1] for i in zip(h3[30], h3[30])))
print('dot sim btwn h3 rows 50/50: ', sum(i[0] * i[1] for i in zip(h3[50], h3[50])))

print('len bin train: ',len(bs_train))
print('len bin test: ',len(bs_test))
print('bin train 0: ', bs_train[0])
print('dot sim btwn bin train 0/1: ', sum(i[0] * i[1] for i in zip(bs_train[0], bs_train[1])))
print('dot sim btwn bin train 0/n: ', sum(i[0] * i[1] for i in zip(bs_train[0], bs_train[100])))
print('diff btwn bin train 0/1: ', bs_train[0]-bs_train[1])
print('diff btwn bin train 0/n: ', bs_train[0]-bs_train[100])
print('diff btwn bin test 0/n: ', bs_test[0]-bs_test[100])

'''
n1=100
n2=80
n3=64
o1=1./n1/n1* LA.norm(1./n1* np.matmul(np.transpose(h1),h1) - np.eye(n1))**2
o2=1./n2/n2* LA.norm(1./n2* np.matmul(np.transpose(h2),h2) - np.eye(n2))**2
o3=1./n3/n3* LA.norm(1./n3* np.matmul(np.transpose(h3),h3) - np.eye(n3))**2
'''
o1,m1 = measure_identity(h1)
o2,m2 = measure_identity(h2)
o3,m3 = measure_identity(h3)

print('o1: ', o1)
print('o2: ', o2)
print('o3: ', o3)

sio.savemat('data_train_chosen_bins.mat', {'data_train_chosen_bins':bs_train})
sio.savemat('data_test_chosen_bins.mat', {'data_test_chosen_bins':bs_test})

sio.savemat('data_train_chosen_img.mat', {'data_train_chosen_img':train_data})
sio.savemat('data_test_chosen_img.mat', {'data_test_chosen_img':test_data})

#np.save('m1.npy',m1)
#np.save('m2.npy',m2)
#np.save('m3.npy',m3)

# np.save(arr=np.array(bs_train),file=os.path.join(root_pth,'train_hash.npy'))
# np.save(arr=np.array(bs_test),file=os.path.join(root_pth,'test_hash.npy'))
