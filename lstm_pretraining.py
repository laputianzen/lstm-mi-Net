#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:05:36 2017

@author: PeterTsai
"""

import numpy as np
import tensorflow as tf
import scipy.io
#import functools
import itertools
import time
import os
from os.path import join as pjoin

# LSTM-autoencoder
from LSTMAutoencoder import *

fold = 0 #0~4
#FLAGS
FLAGS = lambda: None
FLAGS.pretrain_batch_size = 2
FLAGS.exp_dir = 'experiment/Adam'
FLAGS.summary_dir = FLAGS.exp_dir + '/summaries'
FLAGS._ckpt_dir   = FLAGS.exp_dir + '/model'
FLAGS._confusion_dir = FLAGS.exp_dir + '/confusionMatrix'
FLAGS._result_txt = FLAGS.exp_dir + '/final_result.txt'
FLAGS.flush_secs  = 120
FLAGS.pretraining_epochs = 600
FLAGS.finetune_batch_size = 1
FLAGS.finetuning_epochs_epochs = 200

FLAGS.pre_layer_learning_rate = []   
FLAGS.keep_prob = 1.0


# Constants
#player_num = 5
hidden_num = 256
max_step_num = 450
elem_num = 2
iteration = 50
activation_lstm_name = 'softmax' # default tanh
activation_lstm = tf.nn.softmax
# reset tf graph
tf.reset_default_graph()


# placeholder list
"""
p_input = [batch*5,time step, input_feature]
"""
p_input = tf.placeholder(tf.float32, [None, max_step_num, elem_num]) 
seqlen = tf.placeholder(tf.int32,[None])
p_inputs= tf.transpose(p_input, perm=[1,0,2])

cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True, activation=activation_lstm)
with tf.name_scope("ae_lstm"):
    ae = LSTMAutoencoder(hidden_num, p_inputs, seqlen, cell=cell, decode_without_input=True)
    
writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
                                 'lstm_pre_training'),tf.get_default_graph())
writer.close()   
# load dataset
pre_segment_fold_file = 'raw/split/tactic_bagIdxSplit5(1).mat'
cross_fold = scipy.io.loadmat(pre_segment_fold_file)
testIdx_all_fold = cross_fold['test_bagIdx']
testIdx_fold = testIdx_all_fold[0][fold][0]

trajs = scipy.io.loadmat('raw/S_fixed.mat')
S = trajs['S']
numVid, player_num = S.shape

trainIdx_fold = np.arange(numVid)
trainIdx_fold = np.setdiff1d(trainIdx_fold,testIdx_fold)
num_train = len(trainIdx_fold)


seqLen = np.array([S[seq,0].shape[0] for seq in range(len(S))])
XX = []
for p in range(player_num):
    XX.append(seqLen)
seqLenMatrix = np.stack(XX,axis=1)

#zeros padding in traj
padS = []
for v in range(S.shape[0]):
    #print("video:",v)
    vs = np.stack(S[v,:],axis=0)
    #print("before padding:",vs.shape) 
    npad = ((0,0),(0,max_step_num-S[v,0].shape[0]),(0,0))
    pad = np.pad(vs, pad_width = npad, mode='constant', constant_values=0)
    #print("before padding:",pad.shape)
    padS.append(pad)
    
padS = np.stack(padS,axis=0)    


"""
training
"""
loss_queue=[]
sess = tf.Session()

vars_to_init = tf.global_variables()
saver = tf.train.Saver(vars_to_init)

_pretrain_model_dir = '{0}/{1}/lstm_ae/{3}/hidden{2}'.format(FLAGS._ckpt_dir,fold+1,hidden_num,activation_lstm_name)
if not os.path.exists(_pretrain_model_dir):
    os.makedirs(_pretrain_model_dir)
model_ckpt = _pretrain_model_dir  + '/model.ckpt' 
if os.path.isfile(model_ckpt+'.meta'):
    #tf.reset_default_graph()
    print("|---------------|---------------|---------|----------|")
    saver.restore(sess, model_ckpt)
    for v in vars_to_init:
        print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))
else:
    #with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
  
    for i in range(iteration):
        start_time = time.time()
        perm = np.arange(num_train)
        np.random.shuffle(perm)

      
        train_loss = 0.0
        for v in range(num_train): # one video at a time
            player_perm = np.arange(player_num)
            np.random.shuffle(player_perm)            
            random_sequences = padS[perm[v],player_perm]
            batch_seqlen = np.reshape(seqLenMatrix[perm[v],:],(-1))
            loss_val, _ = sess.run([ae.loss, ae.train],  feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
            train_loss += loss_val
            if i % 10 == 0:
                print("iter %d, vid %d: %f" % (i+1, v+1, loss_val))
        print("iter %d:" %(i+1))
        print("train loss: %f" %(train_loss/num_train))        
      
        #test_loss = 0.0
        #for v in range(num_test):
        random_sequences = np.reshape(padS[testIdx_fold,:],(-1,max_step_num,elem_num))
        batch_seqlen = np.reshape(seqLenMatrix[testIdx_fold,:],(-1))
        test_loss = sess.run(ae.loss,  feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
        #test_loss += loss_val
        print("iter %d:" %(i+1))
        print("test loss: %f" %(test_loss))
        time.sleep(2)
      
        loss_queue.append((train_loss/num_train, test_loss))
        duration = time.time() - start_time
        print("duration: %f s" %(duration))
        
    save_path = saver.save(sess, model_ckpt)
    print("Model saved in file: %s" % save_path)#