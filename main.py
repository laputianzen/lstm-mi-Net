#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:14:30 2017

@author: PeterTsai
"""

import numpy as np
import tensorflow as tf
import scipy.io
#import functools
import itertools
import time
import os
import sys
from os.path import join as pjoin

sys.path.append("./codes")
# LSTM-autoencoder
from LSTMAutoencoder import *
import miNet

tf.reset_default_graph()

fold = 0 #0~4
# Constants
#player_num = 5
hidden_num = 256
max_step_num = 450
elem_num = 2
iteration = 2 #500
activation_lstm_name = 'relu6' # default tanh
activation_lstm = tf.nn.relu6

#FLAGS
FLAGS = lambda: None
FLAGS.pretrain_batch_size = 2
FLAGS.exp_dir = 'experiment/Adam'
FLAGS.summary_dir = FLAGS.exp_dir + '/summaries'
FLAGS._ckpt_dir   = FLAGS.exp_dir + '/model'
FLAGS._confusion_dir = FLAGS.exp_dir + '/confusionMatrix'
FLAGS._result_txt = FLAGS.exp_dir + '/final_result.txt'
FLAGS.flush_secs  = 120
FLAGS.pretraining_epochs = 6#600
FLAGS.finetune_batch_size = 2
FLAGS.finetuning_epochs_epochs = 200

FLAGS.pre_layer_learning_rate = []   
FLAGS.keep_prob = 1.0

cross_fold = scipy.io.loadmat('raw/split/tactic_bagIdxSplit5(1).mat')
testIdx_fold = cross_fold['test_bagIdx']
testIdx_fold = testIdx_fold[0][fold][0]
num_test = len(testIdx_fold)
FLAGS.dataset = lambda: None
FLAGS.dataset.testIdx = testIdx_fold

#vid_str = input('Select test video index in fold 1:')
#vid_idx = np.int8(fold1[0][int(vid_str)])
trajs = scipy.io.loadmat('raw/S_fixed.mat')
S = trajs['S']
print('video number:{0}'.format(len(S)))
numVid, player_num = S.shape
FLAGS.dataset.dataTrajOrigin = S

seqLen = np.array([S[seq,0].shape[0] for seq in range(len(S))])
sortIdx = np.argsort(seqLen)
sortSeqLen = np.sort(seqLen)

XX = []
for p in range(5):
    XX.append(seqLen)
seqLenMatrix = np.stack(XX,axis=1)
FLAGS.dataset.seqLenMatrix = seqLenMatrix

padS = []#np.zeros((S.shape[0],S.shape[1],max_step_num ,elem_num))

for v in range(S.shape[0]):
    #print("video:",v)
    vs = np.stack(S[v,:],axis=0)
    #print("before padding:",vs.shape) 
    npad = ((0,0),(0,max_step_num-S[v,0].shape[0]),(0,0))
    pad = np.pad(vs, pad_width = npad, mode='constant', constant_values=0)
    #print("before padding:",pad.shape)
    padS.append(pad)
    
padS = np.stack(padS,axis=0)    
FLAGS.dataset.dataTraj = padS

trainIdx_fold = np.arange(numVid)
trainIdx_fold = np.setdiff1d(trainIdx_fold,testIdx_fold)
num_train = len(trainIdx_fold)
FLAGS.dataset.trainIdx = trainIdx_fold

#batch_xs = np.stack(S[0,:],axis=0)
#B = np.pad(batch_xs, ((0,0),(0,229),(0,0)), 'constant')

C53_combs = list(itertools.combinations([0,1,2,3,4],3))
C52_combs = list(itertools.combinations([0,1,2,3,4],2))
C55_combs = list(itertools.combinations([0,1,2,3,4],5))

# placeholder list
p_input = tf.placeholder(tf.float32, [None, max_step_num, elem_num]) #[batch*5,dynamic step, input_feature]
seqlen = tf.placeholder(tf.int32,[None])
p_inputs= tf.transpose(p_input, perm=[1,0,2])

FLAGS.lstm = lambda:None
FLAGS.lstm.p_input = p_input
FLAGS.lstm.seqlen = seqlen

## cell should be in ae_lstm !!! fix in the future
cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True, activation=activation_lstm)
with tf.name_scope("ae_lstm"):
    ae = LSTMAutoencoder(hidden_num, p_inputs, seqlen, cell=cell, decode_without_input=True)

writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
                                  'lstm_pre_training'),tf.get_default_graph())
writer.close()

def genearateTFCombs(tf_output,np_combs,player_num):
    tf_batch_size = tf.divide(tf.shape(tf_output)[0],tf.constant(player_num))
    index = tf.range(0, tf_batch_size) * player_num
    CC = []
    k = len(np_combs[0])
    for c in range(len(np_combs)*k):
        CC.append(index)
    CC = tf.stack(CC,axis=0)
    CC = tf.transpose(tf.reshape(CC,[len(np_combs),k,-1]),perm=(2,0,1))
    t = tf.stack([tf.cast(tf_batch_size,tf.int32), 1,1])
    tf_np_combs = tf.tile(tf.expand_dims(tf.constant(np_combs),axis=0),t)
#    np_cc = np.stack(np_combs,axis=0)
#    arrays = [np_cc  for _ in range(np_batch_size)]
#    np_cc = np.stack(arrays, axis=0).astype(np.int32)
    combs = tf.add(tf_np_combs,tf.cast(CC,tf.int32))
    return combs

#C53_input = tf.gather(ae.last_output,combs)    
bb = 4
A = padS[0:bb,:,:,:]
A1 = np.reshape(padS[0:bb,:,:,:],(-1,450,2))
batch_seqlen = np.reshape(seqLenMatrix[0:bb,:],(-1))
with tf.name_scope("nchoosek_group_inputs"):
    #tf_C53_combs = genearateTFCombs(C53_combs,ae.last_output,player_num,np_batch_size)
    tf_C53_combs = genearateTFCombs(ae.last_output,C53_combs,player_num)
    tf_C52_combs = genearateTFCombs(ae.last_output,C52_combs,player_num)
    tf_C55_combs = genearateTFCombs(ae.last_output,C55_combs,player_num)
    #cmb = tf.tile(tf.expand_dims(tf.constant(C53_combs),axis=0),[bb,1,1])
#    with tf.Session() as sess:     
#        sess.run(tf.global_variables_initializer())
#        cmb_result = sess.run(cmb, 
#             feed_dict={p_input:A1,seqlen: batch_seqlen})        
#        ss, ii, cc, cbb, c53_cc = sess.run([batch_size,index,CC,combs,C53_input], 
#             feed_dict={p_input:A1,seqlen: batch_seqlen})
#        hidden = sess.run(ae.last_output, 
#             feed_dict={p_input:A1,seqlen: batch_seqlen})
#        c1 = c53_cc[0,:,:,:]
    C53_input = tf.gather(ae.last_output,tf_C53_combs)
    C52_input = tf.gather(ae.last_output,tf_C52_combs)
    C55_input = tf.gather(ae.last_output,tf_C55_combs)

    # [batch_num, nchoosek_num, k_num, feature_dim]
    C53_input_merge = tf.reduce_mean(C53_input,axis=2,name='C53')
    C52_input_merge = tf.reduce_mean(C52_input,axis=2,name='C52')
    C55_input_merge = tf.reduce_mean(C55_input,axis=2,name='C55')
    nchoosek_inputs = [C53_input_merge, C52_input_merge, C55_input_merge]

#    C53_input_merge_reshape = tf.reshape(C53_input_merge,[len(C53_combs),hidden_num])
#    C52_input_merge_reshape = tf.reshape(C52_input_merge,[len(C52_combs),hidden_num])
#    C55_input_merge_reshape = tf.reshape(C55_input_merge,[len(C55_combs),hidden_num])
#    group_inputs = [C53_input_merge_reshape, C52_input_merge_reshape, C55_input_merge_reshape]

"""
pre-training cycle
"""
loss_queue=[]
sess = tf.Session()

vars_to_init = tf.global_variables()
saver = tf.train.Saver(vars_to_init)

_pretrain_model_dir = '{0}/{1}/lstm_ae/{3}/hidden{2}/iter{4}'.format(
        FLAGS._ckpt_dir,fold+1,hidden_num,activation_lstm_name,iteration)
if not os.path.exists(_pretrain_model_dir):
    os.makedirs(_pretrain_model_dir)
model_ckpt = _pretrain_model_dir  + '/model.ckpt' 
FLAGS.lstm_ae_model_ckpt = model_ckpt
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
                print("iter %d, vid %d: %f" % (i+1, perm[v]+1, loss_val))
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

"""
generate intermediate feature of training data
"""
C53_data = []
C52_data = []
C55_data = []
for v in range(num_train):
    random_sequences = padS[trainIdx_fold[v],:]
    batch_seqlen = np.reshape(seqLenMatrix[trainIdx_fold[v],:],(-1))
    C53_merge = sess.run(C53_input_merge, 
             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
    C52_merge = sess.run(C52_input_merge, 
             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})    
    C55_merge = sess.run(C55_input_merge, 
             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
    C53_data.append(C53_merge)
    C52_data.append(C52_merge)
    C55_data.append(C55_merge)

_intermediate_feature_dir = _pretrain_model_dir  + '/tempData'
if not os.path.exists(_intermediate_feature_dir):
    os.makedirs(_intermediate_feature_dir)
C53_data = np.concatenate(C53_data,axis=0)
C52_data = np.concatenate(C52_data,axis=0)
C55_data = np.concatenate(C55_data,axis=0)
np.save(_intermediate_feature_dir + '/C53_fold{0}_train.npy'.format(fold+1),C53_data)
np.save(_intermediate_feature_dir + '/C52_fold{0}_train.npy'.format(fold+1),C52_data)
np.save(_intermediate_feature_dir + '/C55_fold{0}_train.npy'.format(fold+1),C55_data)

C53_data = []
C52_data = []
C55_data = []
# generate intermediate feature of training data    
for v in range(num_test):
    random_sequences = padS[testIdx_fold[v],:]
    batch_seqlen = np.reshape(seqLenMatrix[testIdx_fold[v],:],(-1))
    C53_merge = sess.run(C53_input_merge, 
             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
    C52_merge = sess.run(C52_input_merge, 
             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})    
    C55_merge = sess.run(C55_input_merge, 
             feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
    C53_data.append(C53_merge)
    C52_data.append(C52_merge)
    C55_data.append(C55_merge)

_intermediate_feature_dir = _pretrain_model_dir  + '/tempData'
if not os.path.exists(_intermediate_feature_dir):
    os.makedirs(_intermediate_feature_dir)
C53_data = np.concatenate(C53_data,axis=0)
C52_data = np.concatenate(C52_data,axis=0)
C55_data = np.concatenate(C55_data,axis=0)
np.save(_intermediate_feature_dir + '/C53_fold{0}_test.npy'.format(fold+1),C53_data)
np.save(_intermediate_feature_dir + '/C52_fold{0}_test.npy'.format(fold+1),C52_data)
np.save(_intermediate_feature_dir + '/C55_fold{0}_test.npy'.format(fold+1),C55_data)
#for v in range(num_test):
#    random_sequences = np.stack(S[testIdx_fold1[v],:],axis=0)
#    batch_seqlen = random_sequences.shape[1]

#C53 = sess.run(C53_input,  feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
#C53_merge = sess.run(C53_input_merge,  feed_dict={p_input:random_sequences,seqlen: batch_seqlen})

#C53_input_merge_fixed = tf.Variable(tf.identity(C53_input_merge_reshape),
#                      name="C53_input_merge_fixed", trainable=False)


def createPretrainShape(num_input,num_output,num_hidden_layer):
    reduce_dim = round((num_input-num_output)/(num_hidden_layer+1))
    shape = np.zeros(num_hidden_layer+2,dtype=np.int32)
    shape[0] = num_input
    shape[num_hidden_layer+1] = num_output
    for l in range(num_hidden_layer):
        shape[l+1] = shape[l] - reduce_dim
        
    return shape

num_output = 16
num_hidden_layer = 1
fold = 0
pretrain_shape = createPretrainShape(hidden_num,num_output,num_hidden_layer)
print(pretrain_shape)  
FLAGS.tacticName =['F23','EV','HK','PD','PT','RB','SP','WS','WV','WW']
FLAGS.C5k_CLASS = [[0,1,2,3,5,7],[6,9],[4,8]]
FLAGS.k = [3,2,5]
FLAGS.playerMap = [[[1,1,1,0,0],[1,1,0,1,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1],
                       [1,0,0,1,1],[0,1,1,1,0],[0,1,1,0,1],[0,1,0,1,1],[0,0,1,1,1]],
                   [[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],
                       [0,1,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,0,1,1]],
                   [[1,1,1,1,1]]];
                    
for h in range(num_hidden_layer+1):
    FLAGS.pre_layer_learning_rate.extend([0.001])#GD[0.01,0.01]
FLAGS.optim_method = tf.train.AdamOptimizer
FLAGS.supervised_learning_rate = 0.001
#instNet_shape = [1040,130,10,1] #[1040,10,1]
instNet_shape = np.array([np.append(pretrain_shape,len(FLAGS.C5k_CLASS[0])),
                          np.append(pretrain_shape,len(FLAGS.C5k_CLASS[1])),
                          np.append(pretrain_shape,len(FLAGS.C5k_CLASS[2]))],
                         np.int32)
FLAGS._intermediate_feature_dir = _intermediate_feature_dir
print(instNet_shape)
num_inst = np.array([10,10,1],np.int32) # 5 choose 3 key players, 5 choose 2 key players, 5 choose 3 key players 
miList = miNet.main_unsupervised(instNet_shape,fold,FLAGS)
miNet.main_supervised(miList,num_inst,nchoosek_inputs,fold,FLAGS)
        

#  new_variables_names = [v.name for v in tf.trainable_variables()]
#  all_variables_names = [v.name for v in tf.global_variables()]
#  
#  graph_def=sess.graph_def
##  with tf.Graph().as_default() as g:
##      input = tf.placeholder(tf.float32)
##      output = tf.identity(input, name='output')
##      graph_def = g.as_graph_def()
#  print("Before:")
#  nodes = [n.name for n in graph_def.node]
#  print(nodes)
#  print("After:")
#  prune_nodes = [n.name for n in tf.graph_util.remove_training_nodes(graph_def).node]
#  print(prune_nodes)
#  node_diff = list(set(nodes) - set(prune_nodes))
#  optim_nodes = [var for var in prune_nodes if ('beta' in var or 'Adam' in var)]
#  
#  vars_to_freeze = [var.name for var in tf.trainable_variables() if ('encoder' in var.name or 'rnn' in var.name)]
#  const_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph_def,vars_to_freeze)
#  [XX.name for XX in tf.get_default_graph().node]
#  tf.graph_util.remove_training_nodes(tf.get_default_graph())
#  
#  new_variables_names = [v.name for v in tf.trainable_variables()]
#  all_variables_names = [v.name for v in tf.global_variables()]
  
  



#seqLen = tf.placeholder(tf.int32, [None,])
#tmpList = list()
#for i in range(5):
#    with tf.variable_scope("group") as scope:
#        if i == 0:
#            tmpList.append(ae.last_output)
#            scope.reuse_variables()
#        else:
#            tmpList.append(ae.last_output)
        





