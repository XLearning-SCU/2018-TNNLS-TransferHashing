'''
Deep Hashing with Multilayer Perceptron implementation using TensorFlow library.
'''

from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from random import shuffle
from multilayer_perceptron import multilayer_perceptron
from multilayer_perceptron import multilayer_perceptron_aux
import logging
import time
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def my_shuffle(x):
    idx = list(range(len(x)))
    shuffle(idx)
    x_tmp = []
    for i in idx:
        x_tmp.append(x[i])
    return x_tmp

def _configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
    return tf.train.exponential_decay(learning_rate_init,
                                      global_step,
                                      decay_steps,
                                      learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name', default=None, required=True)
parser.add_argument('--use_aux', help='dataset name',  default=True)
args = parser.parse_args()

datasets = ['bbc','reuters']
dataset = args.dataset
USE_TXT = bool(args.use_aux)

print('using %s\nwith pi: %s'%(dataset,USE_TXT))
logging.basicConfig(filename='log_%s.log'%dataset,level=logging.DEBUG)
ckpt_save_folder = 'save_%s'%dataset
if os.path.isdir(ckpt_save_folder) == False:
    try:
        os.mkdir(ckpt_save_folder)
    except:
        raise ('ckpt_save_folder not exist')

# ntwk_params = [n_input, n_hidden_1, n_hidden_2, n_bins]
if dataset == 'bbc':
    # NETWORK PARAMETERS
    ntwk_params = [411,2048,2048,64]
    ntwk_params_aux = [412,256,256,64]

    # TRAINING PARAMETERS
    learning_rate_init = 0.01
    training_epochs = 400
    batch_size = 200
    display_step = 1
    save_step = 30
    num_epochs_per_decay = 10
    learning_rate_decay_factor = 0.8

    # DEFINE LOSS AND OPTIMIZER
    lmd_1 = 15.  # quant
    lmd_2 = 1.  # q_err
    lmd_3 = 1.  # weight orth,
    lmd_4 = 10.  # similarity matrix
    lmd_5 = 1e-5  # regularizer & regularizer_aux
    lmd_6 = 1e-5  # regularizer_aux
elif dataset == 'reuters':
    # NETWORK PARAMETERS
    ntwk_params = [1267,4096,2048,64]
    ntwk_params_aux = [784,1024,256,64]

    # TRAINING PARAMETERS
    learning_rate_init = 0.01
    training_epochs = 200
    batch_size = 200
    display_step = 1
    save_step = 4
    num_epochs_per_decay = 5
    learning_rate_decay_factor = 0.3

    # DEFINE LOSS AND OPTIMIZER
    lmd_1 = 15.      #quant
    lmd_2 = 1.      #q_err
    lmd_3 = 1.      #weight orth,
    lmd_4 = 10.      #similarity matrix
    lmd_5 = 1e-5    #regularizer & regularizer_aux
    lmd_6 = 1e-5    #regularizer_aux
else:
    raise ('dataset invalid!')

data_all_raw = np.load('data_train_chosen_%s.npy'%dataset)
TRAIN_NUM = len(data_all_raw)
data_train_all = []
for i in range(TRAIN_NUM):
    data_train_all.append([data_all_raw[i][0],data_all_raw[i][1]])
data_all_raw = []
data_train = data_train_all
TOTAL_NUM_SAMPLES = len(data_train)

# Network Parameters
n_input = ntwk_params[0]
n_input_aux = ntwk_params_aux[0]
n_bins = ntwk_params[3]

# TF GRAPH INPUT
x = tf.placeholder("float", [None, n_input],name='input_x')
x_txt = tf.placeholder('float',[None,n_input_aux],name='n_input_aux')
x_sim = tf.placeholder('float',[batch_size,batch_size], name='input_x_sim')

# CONSTRUCT MODEL
reals, bins, reg, orth = multilayer_perceptron(x=x, params=ntwk_params)
out_txt, reg_txt = multilayer_perceptron_aux(x=x_txt, params=ntwk_params_aux)

zb_err = 1.0*tf.subtract(bins,reals)
quant = 0.5*tf.square(tf.norm(zb_err))
sim = 0.5*tf.square(tf.norm(tf.subtract( (1./n_bins)*tf.matmul(a=reals,b=reals,transpose_a=False, transpose_b=True), x_sim)))

if USE_TXT == True:
    quant_aprox_err = 0.5 * tf.square(tf.norm(tf.subtract(zb_err, out_txt)))
    cost = tf.add_n([lmd_1*quant, \
                     lmd_2*quant_aprox_err, \
                     # lmd_3*orth, \
                     lmd_4*sim, \
                     lmd_5*reg_txt])
else:
    cost = tf.add_n([lmd_1*quant, \
                     # lmd_3*orth, \
                     lmd_4*sim, \
                     lmd_5*reg])

global_step = tf.contrib.slim.create_global_step()
learning_rate = _configure_learning_rate(TOTAL_NUM_SAMPLES, global_step)

# tflearn.optimizers.Momentum (learning_rate=0.001, \
#                              momentum=0.9, \
#                              lr_decay=0.0, \
#                              decay_step=100, \
#                              staircase=False, \
#                              use_locking=False, \
#                              name='Momentum')

# optimizer = tflearn.optimizers.RMSProp (learning_rate=0.001, \
#                             decay=0.9, \
#                             momentum=0.0, \
#                             epsilon=1, \
#                             use_locking=False, \
#                             name='RMSProp')

optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9,use_nesterov=False,name='Momentum')
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1)

tvars = tf.trainable_variables()

grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)

optimizer1 = optimizer.apply_gradients(zip(grads, tvars),global_step=global_step)

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# gvs = optimizer.compute_gradients(cost)
# capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
# optimizer1 = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver(max_to_keep=20)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, './%s/mlp'%ckpt_save_folder, global_step=-1)
    # Training cycle
    for epoch in range(training_epochs):

        # SOMEHOW DEFAULT SHUFFLE HAS SOME BUGS AND DIDN'T WORK AS EXPECTED
        data_train = my_shuffle(data_train)

        # MONITOR PARAMETER
        avg_cost = 0.
        avg_q = 0.
        avg_s = 0.
        avg_q_aprx = 0.
        avg_prc_time = 0.
        avg_opt_time = 0.
        total_batch = int(TOTAL_NUM_SAMPLES/batch_size)

        # LOOP OVER ALL BATCHES
        if USE_TXT == True:
            for i in range(total_batch):
                batch = []
                batch_x = []
                batch_x_txt = []
                batch_x_idx = []

                start = time.time()
                batch = data_train[i*batch_size:(i+1)*batch_size]
                for item in batch:
                    batch_x.append(np.squeeze(item[0]))
                    batch_x_txt.append(np.squeeze(item[1]))
                batch = []
                # CALCULATE COSINE_SIMILARITY FOR CURRENT BATCH
                batch_x_sim = cosine_similarity(batch_x,batch_x)
                end = time.time()
                prc_time = end - start

                start = time.time()

                # RUN OPTIMIZATION OP (BACKPROP) AND COST OP (TO GET LOSS VALUE)
                _, c, q, q_aprx, s, r, r1,lr = sess.run([optimizer1, \
                                                        cost, \
                                                        lmd_1 * quant, \
                                                        lmd_2 * quant_aprox_err, \
                                                        lmd_4 * sim, \
                                                        lmd_5 * reg, \
                                                        lmd_6 * reg_txt, \
                                                        learning_rate], \
                                                       feed_dict={x: batch_x, x_txt:batch_x_txt, x_sim:batch_x_sim})
                end = time.time()
                opt_time = end - start

                # COMPUTE AVERAGE LOSS
                avg_cost += c / total_batch
                avg_q += q / total_batch
                avg_q_aprx += q_aprx / total_batch
                avg_s += s / total_batch
                avg_prc_time += prc_time / total_batch
                avg_opt_time += opt_time / total_batch
            # DISPLAY LOGS PER EPOCH STEP
            if epoch % display_step == 0:
                s1 = "Epoch %04d | lr: %06f | cost: %09f | quant %09f | q_aprx %09f |sim %09f | reg %09f | avg_prc_time %05f | avg_opt_time %05f |" \
                     % (epoch+1, lr, avg_cost, avg_q, avg_q_aprx, avg_s, r, avg_prc_time,avg_opt_time)
                logging.info(s1)
                print(s1)
        else:
            for i in range(total_batch):
                batch = []
                batch_x = []
                batch_x_idx = []
                batch_x_txt = []
                batch_x_sim = []

                start = time.time()
                batch = data_train[i*batch_size:(i+1)*batch_size]
                for item in batch:
                    batch_x.append(np.squeeze(item[0]))
                batch = []
                # CALCULATE COSINE_SIMILARITY FOR CURRENT BATCH
                batch_x_sim = cosine_similarity(batch_x,batch_x)
                end = time.time()
                prc_time = end - start

                start = time.time()
                # RUN OPTIMIZATION OP (BACKPROP) AND COST OP (TO GET LOSS VALUE)
                _, c, q, s, r, lr = sess.run([optimizer1, \
                                                        cost, \
                                                        lmd_1 * quant, \
                                                        # lmd_2 * max_var_item, \
                                                        # lmd_3 * orth, \
                                                        lmd_4 * sim, \
                                                        lmd_5 * reg, \
                                                        learning_rate], \
                                                       feed_dict={x: batch_x, x_sim: batch_x_sim})
                end = time.time()
                opt_time = end - start

                # COMPUTE AVERAGE LOSS
                avg_cost += c / total_batch
                avg_q += q / total_batch
                avg_s += s / total_batch
                avg_prc_time += prc_time / total_batch
                avg_opt_time += opt_time / total_batch
            # DISPLAY LOGS PER EPOCH STEP
            if epoch % display_step == 0:
                s1 = "Epoch %04d | lr: %06f | cost: %09f | quant %09f | sim %09f | reg %09f | avg_prc_time %05f | avg_opt_time %05f |" \
                     % (epoch+1, lr, avg_cost, avg_q, avg_s, r, avg_prc_time,avg_opt_time)
                logging.info(s1)
                print(s1)

        if epoch % save_step == 0:
            saver.save(sess, './%s/mlp'%ckpt_save_folder, global_step=epoch)

    print("Optimization Finished!")
    saver.save(sess, './%s/mlp'%ckpt_save_folder, global_step=training_epochs)






