#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 01:32:17 2017

@author: PeterTsai
"""

from __future__ import division
from __future__ import print_function
import time
import os
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
import pandas as pd 


import readmat
#from flags import FLAGS

class miNet(object):
    
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"
    _inputs_str = "x{0}"
    _instNets_str = "I{0}"

    def __init__(self, shape, sess):
        """Autoencoder initializer

        Args:
            shape: list of ints specifying
                  num input, hidden1 units,...hidden_n units, num logits
        sess: tensorflow session object to use
        """
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__num_hidden_layers = len(self.__shape) - 2

        self.__variables = {}
        self.__sess = sess

        self._setup_instNet_variables()
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers
    
    @property
    def session(self):
        return self.__sess
    
    def __getitem__(self, item):
        """Get autoencoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
            item: string, variables internal name
        Returns:
            Tensorflow variable
        """
        return self.__variables[item]
    
    def __setitem__(self, key, value):
        """Store a tensorflow variable

        NOTE: Don't call this explicity. It should
        be used only internally when setting up
        variables.

        Args:
            key: string, name of variable
            value: tensorflow variable
        """
        self.__variables[key] = value
    
    def _setup_instNet_variables(self):
        with tf.name_scope("InstNet_variables"):
            for i in range(self.__num_hidden_layers + 1):
                # Train weights
                name_w = self._weights_str.format(i + 1)
                w_shape = (self.__shape[i], self.__shape[i + 1])
                #a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
                #w_init = tf.random_uniform(w_shape, -1 * a, a)
                w_init = tf.random_normal(w_shape,stddev=0.01)
                self[name_w] = tf.Variable(w_init, name=name_w, trainable=True)
                # Train biases
                name_b = self._biases_str.format(i + 1)
                b_shape = (self.__shape[i + 1],)
                b_init = tf.zeros(b_shape) + 0.01
                self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

                if i < self.__num_hidden_layers:
                    # Hidden layer fixed weights (after pretraining before fine tuning)
                    self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                        name=name_w + "_fixed", trainable=False)
                    
                    # Hidden layer fixed biases
                    self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                        name=name_b + "_fixed", trainable=False)
                    
                    # Pretraining output training biases
                    name_b_out = self._biases_str.format(i + 1) + "_out"
                    b_shape = (self.__shape[i],)
                    b_init = tf.zeros(b_shape) + 0.01
                    self[name_b_out] = tf.Variable(b_init, trainable=True, name=name_b_out)
                    
    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]
    
    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]
    
    def get_variables_to_init(self, n):
        """Return variables that need initialization

        This method aides in the initialization of variables
        before training begins at step n. The returned
        list should be than used as the input to
        tf.initialize_variables

        Args:
            n: int giving step of training
        """
        assert n > 0
        assert n <= self.__num_hidden_layers + 1

        vars_to_init = [self._w(n), self._b(n)]

        if n <= self.__num_hidden_layers:
            vars_to_init.append(self._b(n, "_out"))
            
        if 1 < n <= self.__num_hidden_layers:
            vars_to_init.append(self._w(n - 1, "_fixed"))
            vars_to_init.append(self._b(n - 1, "_fixed"))
            
        return vars_to_init
    
    @staticmethod
    def _activate(x, w, b, transpose_w=False, acfun=None, keep_prob=1):
        dropout_out = tf.nn.dropout(x,keep_prob)  
        if acfun is not None:
            y = acfun(tf.nn.bias_add(tf.matmul(dropout_out, w, transpose_b=transpose_w), b))
        else:
            y = tf.nn.bias_add(tf.matmul(dropout_out, w, transpose_b=transpose_w), b)
                         
        return y
    
    def pretrain_net(self, input_pl, n, is_target=False):
        """Return net for step n training or target net
        
        Args:
            input_pl:  tensorflow placeholder of AE inputs
            n:         int specifying pretrain step
            is_target: bool specifying if required tensor
                       should be the target tensor
        Returns:
            Tensor giving pretraining net or pretraining target
        """
        assert n > 0
        assert n <= self.__num_hidden_layers

        last_output = input_pl
        for i in range(n - 1):
            if i == self.__num_hidden_layers+1:
                acfun = tf.sigmoid
            else:
                acfun = tf.nn.relu
            #acfun = "sigmoid"
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")
            
            last_output = self._activate(last_output, w, b, acfun=acfun)         
            
        if is_target:
            return last_output
        
        if n == self.__num_hidden_layers+1:
            acfun = tf.sigmoid
        else:
            acfun = tf.nn.relu       
        last_output = self._activate(last_output, self._w(n), self._b(n), acfun=acfun)
        
        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                         transpose_w=True, acfun=acfun)
        out = tf.maximum(out, 1.e-9)
        out = tf.minimum(out, 1 - 1.e-9)
        return out
    
    def single_instNet(self, input_pl, dropout):
        """Get the supervised fine tuning net

        Args:
            input_pl: tf placeholder for ae input data
        Returns:
            Tensor giving full ae net
        """
        last_output = input_pl
        
        for i in range(self.__num_hidden_layers + 1):
            if i+1 == self.__num_hidden_layers+1:
                acfun = tf.log_sigmoid
                dropout = 1.0
            else:
                acfun = tf.nn.relu
            #acfun = "sigmoid" acfun = "relu"            
            # Fine tuning will be done on these variables
            w = self._w(i + 1)
            b = self._b(i + 1)
            
            last_output = self._activate(last_output, w, b, acfun=acfun, keep_prob=dropout)
            
        return last_output
    
    def MIL(self,input_plist, dropout):
        #input_dim = self.shape[0]
        tmpList = list()
        for i in range(int(input_plist.shape[0])):
            name_input = self._inputs_str.format(i + 1)
            #self[name_input] = tf.placeholder(tf.float32,[None, input_dim])
            self[name_input] = input_plist[i]
            
            name_instNet = self._instNets_str.format(i + 1)
            with tf.variable_scope("mil") as scope:
                if i == 0:
                #if scope.reuse == False:
                    self[name_instNet] = self.single_instNet(self[name_input], dropout)
                    #scope.reuse = True
                    scope.reuse_variables()
                else:    
                    self[name_instNet] = self.single_instNet(self[name_input], dropout)
            
            tmpList.append(self[name_instNet])
            #if not i == 0:
                #self["y"]  = tf.concat([self["y"],self[name_instNet]],1)
            #    self["y"]  = [self["y"],self[name_instNet]]
            #else:
            #    self["y"] = self[name_instNet]
        
        self["y"] = tf.stack(tmpList,axis=1)
        self["Y"] =  tf.reduce_max(self["y"],axis=1,name="MILPool")#,keep_dims=True)
        self["maxInst"] = tf.argmax(self["y"],axis=1, name="maxInst")
        
        #batch_size = int(self["y"].shape[0])
        #topInstIdx = tf.reshape(tf.argmax(self["y"],axis=1),[batch_size,1])
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(tf.argmax(self["y"],axis=1)+1,tf.float32),name='key_instance')
        
        #topInstIdx = tf.argmax(self["y"],axis=1)
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(topInstIdx+1,tf.float32),name='key_instance')
        # consider tf.expand_dims to support tf.argmax
        
        #return self["Y"], self["kinst"]
        return self["Y"], self["y"], self["maxInst"]


loss_summaries = {}

def training(loss, learning_rate, loss_key=None, optimMethod=tf.train.AdamOptimizer, var_in_training=None):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage

  Returns:
    train_op: The Op for training.
  """
  if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
  else:
    tf.summary.scalar(loss.op.name, loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
  # Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.AdamOptimizer(learning_rate)
  optimizer = optimMethod(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  if var_in_training is not None:
      train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_in_training)
  else:
      train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step

def loss_x_entropy(output, target):
    """Cross entropy loss
    
    See https://en.wikipedia.org/wiki/Cross_entropy

    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
        """
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                           target_tf),
                             tf.multiply(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                   name='xentropy_mean')


def main_unsupervised(ae_shape,fold,FLAGS):
    #tf.reset_default_graph()
    sess = tf.Session()
   
    aeList = list()
    for a in range(len(ae_shape)):
        aeList.append(miNet(ae_shape[a], sess))
    #aeC53 = miNet(ae_shape[0], sess)
    #aeC52 = miNet(ae_shape[1], sess)
    #aeC55 = miNet(ae_shape[2], sess)
    
    #aeList = [aeC53, aeC52, aeC55]
    

    
    learning_rates = FLAGS.pre_layer_learning_rate
    
#for fold in range(5):
    print('fold %d' %(fold+1))
    for k in range(len(ae_shape)):
#        tactic = FLAGS.tacticName[FLAGS.C5k_CLASS[k][0]]
        #datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d/pretraining' %(fold+1)
#        datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d/' %(fold+1)
#        fileName= '%sZoneVelocitySoftAssign(R=16,s=10)%d_training%d.mat' %(tactic,FLAGS.k[k],fold+1)
#    
#        batch_X, batch_Y, _ = readmat.read(datadir,fileName)
#        num_train = len(batch_Y)
#        strBagShape = "the shape of bags is ({0},{1})".format(batch_Y.shape[0],batch_Y.shape[1])
#        print(strBagShape)   
#        
#        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
#        testFileName= '%sZoneVelocitySoftAssign(R=16,s=10)%d_test%d.mat' %(tactic,FLAGS.k[k],fold+1) 
#        test_X, test_Y, test_label = readmat.read(testdir,testFileName)    
#        strBagShape = "the shape of bags is ({0},{1})".format(test_Y.shape[0],test_Y.shape[1])
#        print(strBagShape)  
        file_str = FLAGS._intermediate_feature_dir + "/C5{0}_fold{1}"
        batch_X = np.load(file_str.format(FLAGS.k[k],fold+1)+'_train.npy')
        test_X = np.load(file_str.format(FLAGS.k[k],fold+1)+'_test.npy')
        num_train = len(batch_X)
        
        print("\nae_shape has %s pretarined layer" %(len(ae_shape[k])-2))
        for i in range(len(ae_shape[k]) - 2):
            n = i + 1
            _pretrain_model_dir = '{0}/{1}/C5{2}/pretrain{3}/'.format(FLAGS._ckpt_dir,fold+1,FLAGS.k[k],n)
            if not os.path.exists(_pretrain_model_dir):
                os.makedirs(_pretrain_model_dir)
            
            with tf.variable_scope("pretrain_{0}_mi{1}".format(n,k+1)):
                input_ = tf.placeholder(dtype=tf.float32,
                                        shape=(None, ae_shape[k][0]),
                                        name='ae_input_pl')
                target_ = tf.placeholder(dtype=tf.float32,
                                         shape=(None, ae_shape[k][0]),
                                         name='ae_target_pl')
                #input_ = inputs[k]
                #target_= inputs[k]
                layer = aeList[k].pretrain_net(input_, n)



                with tf.name_scope("target"):
                    target_for_loss = aeList[k].pretrain_net(target_, n, is_target=True)
                    
                if n == aeList[k].num_hidden_layers+1:
                    loss = loss_x_entropy(layer, target_for_loss)
                else:
                    #loss = tf.sqrt(tf.nn.l2_loss(tf.subtract(layer, target_for_loss)))
                    loss  = tf.sqrt(tf.reduce_mean(tf.square(layer - target_for_loss)))
                        
                vars_to_init = aeList[k].get_variables_to_init(n)
                

                train_op, global_step = training(loss, learning_rates[i], i, 
                                        optimMethod=FLAGS.optim_method, var_in_training=vars_to_init)

                vars_to_init.append(global_step)    
                writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
                                                  'instNet_pre_training'),tf.get_default_graph())
                writer.close()  
                
                # adam special parameter beta1, beta2
                pretrain_vars =  tf.get_collection(tf.GraphKeys.VARIABLES, scope="pretrain_{0}_mi{1}".format(n,k+1))
                optim_vars = [var for var in pretrain_vars if ('beta' in var.name or 'Adam' in var.name)]
#                for var in adam_vars:
#                    vars_to_init.append(var)
                        
                pretrain_test_loss  = tf.summary.scalar('pretrain_test_loss',loss)
                
                saver = tf.train.Saver(vars_to_init)
                model_ckpt = _pretrain_model_dir+ 'model.ckpt'    
            
                if os.path.isfile(model_ckpt+'.meta'):
                    #tf.reset_default_graph()
                    print("|---------------|---------------|---------|----------|")
                    saver.restore(sess, model_ckpt)
                    for v in vars_to_init:
                        print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))
#                text_file = open("Reload.txt", "a")
#                for b in range(len(ae_shape) - 2):
#                    if sess.run(tf.is_variable_initialized(ae._b(b+1))):
#                        #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                        text_file.write("%s with value in [pretrain %s]\n %s\n" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                text_file.close()                    
                else:
                    summary_dir = pjoin(FLAGS.summary_dir, 'fold{0}/mi{1}/pretraining_{2}'.format(fold+1,k+1,n))
                    summary_writer = tf.summary.FileWriter(summary_dir,
                                                           graph_def=sess.graph_def,
                                                           flush_secs=FLAGS.flush_secs)
                    summary_vars = [aeList[k]["biases{0}".format(n)], aeList[k]["weights{0}".format(n)]]
    
                    hist_summarries = [tf.summary.histogram(v.op.name, v)
                                   for v in summary_vars]
                    hist_summarries.append(loss_summaries[i])
                    summary_op = tf.summary.merge(hist_summarries)                    
                    
                    if FLAGS.pretrain_batch_size is None:
                        FLAGS.pretrain_batch_size = batch_X.shape[0]
                    sess.run(tf.variables_initializer(vars_to_init))
                    sess.run(tf.variables_initializer(optim_vars))
                    print("\n\n")
                    print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
                    print("|---------------|---------------|---------|----------|")
        
                    count = 0
                    for epochs in range(FLAGS.pretraining_epochs):
                        perm = np.arange(num_train)
                        np.random.shuffle(perm)
                        for step in range(int(num_train/FLAGS.pretrain_batch_size)):
                            selectIndex = perm[FLAGS.pretrain_batch_size*step:FLAGS.pretrain_batch_size*step+FLAGS.pretrain_batch_size]
                            #for I in range(len(batch_X[0])):
                            #input_feed = batch_X[perm[2*step:2*step+2],i,:]
                            ##target_feed = batch_Y[2*step:2*step+2,1]
                            #target_feed = batch_X[perm[2*step:2*step+2],i,:]
                            #feed_dict = {input_: input_feed,target_: target_feed}
                            #feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
                            input_feed = np.reshape(batch_X[selectIndex,:,:],
                                                    (batch_X[selectIndex,:,:].shape[0]*batch_X[selectIndex,:,:].shape[1],batch_X[selectIndex,:,:].shape[2]))
                            target_feed = input_feed
                            loss_summary, loss_value = sess.run([train_op, loss],
                                                            feed_dict={
                                                                input_: input_feed,
                                                                target_: target_feed,
                                                                })

                            count = count + 1
#                            #if count % 100 == 0:
#                                if count % (10*len(input_feed)*len(batch_X[0])) == 0:
                            if (count-1)*FLAGS.pretrain_batch_size*batch_X.shape[1] % (10*batch_X.shape[1]) ==0:
                                summary_str = sess.run(summary_op, feed_dict={
                                                                input_: input_feed,
                                                                target_: target_feed,
                                                                })
                                summary_writer.add_summary(summary_str, count)
                        #image_summary_op = \
                        #tf.image_summary("training_images",
                        #                 tf.reshape(input_,
                        #                            (FLAGS.batch_size,
                        #                             FLAGS.image_size,
                        #                             FLAGS.image_size, 1)),
                        #                 max_images=FLAGS.batch_size)
        
                        #summary_img_str = sess.run(image_summary_op,
                        #                       feed_dict=feed_dict)
                        #summary_writer.add_summary(summary_img_str)
        
                                output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |"\
                                        .format(step, loss_value, n, epochs + 1)
    
                                print(output)
                        test_input_feed = np.reshape(test_X,(test_X.shape[0]*test_X.shape[1],test_X.shape[2]))
                        test_target_feed = np.reshape(test_X,(test_X.shape[0]*test_X.shape[1],test_X.shape[2]))
                        #test_target_feed = test_Y.astype(np.int32)     
                        loss_summary, loss_value = sess.run([train_op, loss],
                                                            feed_dict={
                                                                    input_: test_input_feed,
                                                                    target_: test_target_feed,
                                                                    })
    
                        pretrain_test_loss_str = sess.run(pretrain_test_loss,
                                                  feed_dict={input_: test_input_feed,
                                                             target_: test_target_feed,
                                                     })                                          
                        summary_writer.add_summary(pretrain_test_loss_str, epochs)
                        print ('epoch %d: test loss = %.3f' %(epochs,loss_value))                           
                    summary_writer.close()         
#                text_file = open("Output.txt", "a")
#                for b in range(len(ae_shape) - 2):
#                    if sess.run(tf.is_variable_initialized(ae._b(b+1))):
#                        #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                        text_file.write("%s with value in [pretrain %s]\n %s\n" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                text_file.close()
                    save_path = saver.save(sess, model_ckpt)
                    print("Model saved in file: %s" % save_path)
                                      
                #input("\nPress ENTER to CONTINUE\n")  
    
        time.sleep(0.5)
                                      
    return aeList

def multiClassEvaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
        """    
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accu  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    error = tf.subtract(1.0,accu)
    return accu, error

def calculateAccu(Y_pred,inst_pred,test_Y,test_label,FLAGS):
          
    
    KP_pred = np.zeros((len(Y_pred),5))
    for bagIdx in range(len(Y_pred)):
        for k in range(len(FLAGS.k)):
            if Y_pred[bagIdx] in FLAGS.C5k_CLASS[k]:
                c = FLAGS.C5k_CLASS[k].index(Y_pred[bagIdx])
                kinst = np.argmax(inst_pred[k][bagIdx,:,c])
                KP_pred[bagIdx] = FLAGS.playerMap[k][kinst]
                
    Y_correct = np.equal(Y_pred,np.argmax(test_Y,1))
    bagAccu = np.sum(Y_correct) / Y_pred.size
    
    y_correct = np.equal(KP_pred[Y_correct,:],test_label[Y_correct,:])
    
    pAccu = np.sum(y_correct) / KP_pred[Y_correct,:].size
    print('bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
    time.sleep(1)
    return bagAccu, pAccu

#text_file = open("final_result.txt", "w")

def main_supervised(instNetList,num_inst,inputs,fold,FLAGS):
    if not os.path.exists(FLAGS._confusion_dir):
        os.mkdir(FLAGS._confusion_dir)
        
    text_file = open(FLAGS._result_txt,"w")
    with instNetList[0].session.graph.as_default():
        sess = instNetList[0].session
        
#        text_file = open("FineTune.txt", "a")
#        for b in range(instNet.num_hidden_layers + 1):
#            if sess.run(tf.is_variable_initialized(instNet._b(b+1))):
#                #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                text_file.write("%s with value before fine-tuning\n %s\n" % (instNet._b(b+1).name, instNet._b(b+1).eval(sess)))
#        text_file.write("\n\n")
#        with tf.Session() as sess1:
#            saver = tf.train.Saver(tf.global_variables())
#            model_ckpt = '{0}/{1}/model_unsp.ckpt'.format(_chkpt_dir,fold+1)
#            save_path = saver.save(sess1, model_ckpt)
#            print("Model saved in file: %s" % save_path)       
#        
#        new = True
#
#        if not os.path.exists('{0}/{1}'.format(_chkpt_dir,fold+1)):
#            os.mkdir('{0}/{1}'.format(_chkpt_dir,fold+1))        
#            
#        model_ckpt = '{0}/{1}/model_sp.ckpt'.format(_chkpt_dir,fold+1)
#    
#        if os.path.isfile(model_ckpt+'.meta'):
#            input_var = None
#            while input_var not in ['yes', 'no']:
#                input_var = input(">>> We found model.ckpt file. Do you want to load it [yes/no]?")
#            if input_var == 'yes':
#                new = False
        bagOuts = []
        instOuts = []
        maxInstOuts = []
#        totalNumInst = np.sum(num_inst)
        instIdx = np.insert(np.cumsum(num_inst),0,0)
#        input_pl = tf.placeholder(tf.float32, shape=(totalNumInst,None,
#                                                instNetList[0].shape[0]),name='input_pl')
        keep_prob_ = tf.placeholder(dtype=tf.float32,
                                           name='dropout_ratio')

        offset = tf.constant(instIdx)
        hist_summaries = []
        for k in range(len(instNetList)):
            with tf.name_scope('C5{0}'.format(FLAGS.k[k])):            
                #out_Y, out_y, out_maxInst = instNetList[k].MIL(input_pl[instIdx[k]:instIdx[k+1]],keep_prob_)
                out_Y, out_y, out_maxInst = instNetList[k].MIL(tf.transpose(inputs[k],perm=(1,0,2)),keep_prob_)
            #bagOuts.append(tf.transpose(out_Y,perm=[1,0]))
            bagOuts.append(out_Y)
            instOuts.append(out_y)
            #maxInstOuts.append(out_maxInst)
            maxInstOuts.append(out_maxInst+offset[k])
            
            hist_summaries.extend([instNetList[k]['biases{0}'.format(i + 1)]
                              for i in range(instNetList[k].num_hidden_layers + 1)])
            hist_summaries.extend([instNetList[k]['weights{0}'.format(i + 1)]
                                   for i in range(instNetList[k].num_hidden_layers + 1)])
    
        hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v)
                              for v in hist_summaries]
        summary_op = tf.summary.merge(hist_summaries)            
        
        #Y = tf.dynamic_stitch(FLAGS.C5k_CLASS,bagOuts)
        Y = tf.concat(bagOuts,1,name='output')
        
        y_maxInsts = tf.concat(maxInstOuts,1, name='maxInsts')
        
#        # regularization (instance intra-similarity)
#        tactic_pred = tf.one_hot(tf.argmax(tf.nn.softmax(Y),axis=1),len(FLAGS.tacticName))
#        pred_related_inst_idx = tf.multiply(tf.cast(y_maxInsts,tf.float32), tactic_pred)
#        #pred_relate_inst_idx = tf.one_hot(tf.cast(pred_related_inst_idx,tf.int32),totalNumInst)    
#        #pred_relate_inst_idx = tf.reduce_max(pred_related_inst_idx,axis=1)
#        pred_relate_inst_idx =  tf.one_hot(tf.cast(tf.reduce_max(pred_related_inst_idx,axis=1),tf.int32),totalNumInst) 
#        input_pl = tf.expand_dims(tf.concat(inputs,0),axis=0)
##        mask_feature = tf.boolean_mask(input_pl,tf.cast(pred_relate_inst_idx,tf.bool))
##        #tf.ones([1,instNetList[0].shape[0]])
##        matrix = []
##        for d in range(instNetList[0].shape[0]):
##            matrix.append(tactic_pred)
##            
##        XX = []
##        mmatrix = tf.transpose(tf.stack(matrix,axis=2),perm=[1,0,2])
##        
###        expand_mask_feature = tf.zeros([len(FLAGS.tacticName),None,instNetList[0].shape[0]])
##        for t in range(len(FLAGS.tacticName)):
##            XX.append(tf.multiply(mmatrix[t],mask_feature))
###            for d in range(instNetList[0].shape[0]):
###                expand_mask_feature[t,:,d] = tf.multiply(tactic_pred[:,t],mask_feature[:,d])
##        expand_mask_feature = tf.stack(XX,axis=0)
##        numEachTactic = tf.reduce_sum(tactic_pred,axis=0)
##        dist = tf.norm(expand_mask_feature,axis=2)
##        mean, variance = tf.nn.moments(dist,[1])
##        variance = tf.divide(variance,numEachTactic)
##        variance = tf.where(tf.is_nan(variance),tf.zeros_like(variance),variance)
#        saver = tf.train.Saver()
#        if new:
        print("")
        print('fold %d' %(fold+1))
        datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d' %(fold+1)
        file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_training%d.mat' %(fold+1)

        _, batch_multi_Y, batch_multi_KPlabel = readmat.multi_class_read(datadir,file_str,num_inst,FLAGS)
        num_train = len(batch_multi_Y)
        strBagShape = "the shape of bags is ({0},{1})".format(batch_multi_Y.shape[0],batch_multi_Y.shape[1])
        print(strBagShape)
        batch_multi_X = FLAGS.dataset.dataTraj[FLAGS.dataset.trainIdx,:]

        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
        test_file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_test%d.mat' %(fold+1) 
        _, test_multi_Y, test_multi_label = readmat.multi_class_read(testdir,test_file_str,num_inst,FLAGS)       
        strBagShape = "the shape of bags is ({0},{1})".format(test_multi_Y.shape[0],test_multi_Y.shape[1])
        print(strBagShape)
        test_multi_X = FLAGS.dataset.dataTraj[FLAGS.dataset.testIdx,:]
      
        if FLAGS.finetune_batch_size is None:
            FLAGS.finetune_batch_size = len(test_multi_Y)
            
        NUM_CLASS = len(FLAGS.tacticName)
        Y_placeholder = tf.placeholder(tf.float32,
                                        shape=(None,NUM_CLASS),
                                        name='target_pl')
        #loss = loss_x_entropy(tf.nn.softmax(Y), tf.cast(Y_placeholder, tf.float32))
        with tf.name_scope('softmax_cross_entory_with_logit'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y,
                                labels=tf.argmax(Y_placeholder,axis=1),name='softmax_cross_entropy'))
        
#        loss_xs = loss
#        beta = FLAGS.beta
#        loss = loss + tf.reduce_sum(variance) * beta
        loss_op = tf.summary.scalar('test_loss',loss)
        #loss = loss_supervised(logits, labels_placeholder)
        
        with tf.name_scope('MultiClassEvaluation'):
            accu, error = multiClassEvaluation(Y, Y_placeholder)
        #train_op, global_step = training(error, FLAGS.supervised_learning_rate, None, optimMethod=FLAGS.optim_method)
#        with tf.name_scope('correctness'):
#            correct =tf.equal(tf.argmax(Y,1),tf.argmax(Y_placeholder,1))
#            error = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
        
        error_op = tf.summary.scalar('test_error',error)
        accu_op = tf.summary.scalar('test_accuracy',accu)

        Y_labels_image = label2image(Y_placeholder)
        #label_op = tf.summary.image('tactic_labels',Y_label_image)
        Y_logits_image = label2image(tf.nn.softmax(Y))
        #label_op = tf.summary.histogram('tactic_labels',Y_placeholder)
        merged = tf.summary.merge([loss_op,error_op,accu_op,summary_op])#,output_op,label_op])
        summary_writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
                                                      'fold{0}/fine_tuning'.format(fold+1)),tf.get_default_graph())
                                                #graph_def=sess.graph_def,
                                                #flush_secs=FLAGS.flush_secs)
        vars_to_init = []
        # initialize lstm variabel         
        vars_to_init.extend(tf.get_collection(tf.GraphKeys.VARIABLES,scope='encoder'))
                                
        for k in range(len(instNetList)):
            instNet = instNetList[k]
            vars_to_init.extend(instNet.get_variables_to_init(instNet.num_hidden_layers + 1))
#        vars_to_init = tf.trainable_variables()
        
        train_op, global_step = training(loss, FLAGS.supervised_learning_rate, None, optimMethod=FLAGS.optim_method,var_in_training=vars_to_init)
        vars_to_init.append(global_step)
        # adam special parameter beta1, beta2
        optim_vars = [var for var in tf.global_variables() if ('beta' in var.name or 'Adam' in var.name)] 
        
        sess.run(tf.variables_initializer(vars_to_init))
        sess.run(tf.variables_initializer(optim_vars))
    
        train_loss  = tf.summary.scalar('train_loss',loss)
        #steps = FLAGS.finetuning_epochs * num_train
        trainIdx = FLAGS.dataset.trainIdx
        seqLenMatrix = FLAGS.dataset.seqLenMatrix
        for epochs in range(FLAGS.finetuning_epochs_epochs):
            perm = np.arange(num_train)
            np.random.shuffle(perm)
            #numPlayer = len(batch_multi_KPlabel[0])
            print("|-------------|-----------|-------------|----------|")
            text_file.write("|-------------|-----------|-------------|----------|\n")
            for step in range(int(num_train/FLAGS.finetune_batch_size)):
                start_time = time.time()
                
               # player_perm = np.arange(numPlayer)
                #np.random.shuffle(player_perm)          
                selectIndex = perm[FLAGS.finetune_batch_size*step:FLAGS.finetune_batch_size*step+FLAGS.finetune_batch_size]
                #input_feed = batch_multi_X[selectIndex,:]
                random_sequences = np.reshape(batch_multi_X[selectIndex,:],(-1,batch_multi_X.shape[2],batch_multi_X.shape[3]))
                batch_seqlen = np.reshape(seqLenMatrix[trainIdx[selectIndex],:],(-1))
                target_feed = batch_multi_Y[selectIndex].astype(np.int32)         
            
                _, loss_value, logit, label= sess.run([train_op, loss, Y, Y_placeholder],
                                        feed_dict={ 
                                                FLAGS.lstm.p_input:random_sequences,
                                                FLAGS.lstm.seqlen: batch_seqlen,
                                                Y_placeholder: target_feed,
                                                keep_prob_: FLAGS.keep_prob
                                        })
    
#                los, m, var, mf, xpmf, maxInst, ta_pred,num_ta, related_inst_idx, relate_inst_idx = sess.run([loss_xs, mean, variance, mask_feature, expand_mask_feature, y_maxInsts, tactic_pred, numEachTactic, pred_related_inst_idx, pred_relate_inst_idx],
#                                                          feed_dict={
#                                                                  FLAGS.lstm.p_input:random_sequences,
#                                                                  FLAGS.lstm.seqlen: batch_seqlen,
#                                                                  Y_placeholder: target_feed,
#                                                                  keep_prob_: FLAGS.keep_prob
#                                                })
            
                duration = time.time() - start_time
                
                #count = epochs*(num_train/FLAGS.batch_size)+step
                # Write the summaries and print an overview fairly often.
                #if step % 10 == 0:
                if step % 10 == 0:
                    # Print status to stdout.
                    #print('Step %d: loss = %.2f (%.3f sec)' % (count, loss_value, duration))
                    print('|   Epoch %d  |  Step %d  |  loss = %.3f | (%.3f sec)' % (epochs+1, step, loss_value, duration))
                    text_file.write('|   Epoch %d  |  Step %d  |  loss = %.3f | (%.3f sec)\n' % (epochs+1, step, loss_value, duration))
                    
#                # Update the events file.
#                input_feed = np.transpose(batch_multi_X, (1,0,2))
#                target_feed = batch_multi_Y.astype(np.int32) 
#                train_loss_str = sess.run(train_loss,
#                                          feed_dict={input_pl: input_feed,
#                                                     Y_placeholder: target_feed,
#                                                     keep_prob_: 1.0
#                                                     })                                          
#                summary_writer.add_summary(train_loss_str, epochs)                    
                    
# =============================================================================
#             bagAccu_merge =[]
#             Y_pred_merge = []
#             inst_pred_merge=[]
#             for v in range(len(test_multi_Y)):
#                 random_sequences =  np.stack(test_multi_X[v,:],axis=0)
#                 batch_seqlen = random_sequences.shape[1]
#                 test_target_feed = test_multi_Y.astype(np.int32)
#                 bagAccu, Y_pred, inst_pred = sess.run([accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],
#                                                        feed_dict={FLAGS.lstm.p_input:random_sequences,
#                                                                   FLAGS.lstm.seqlen: batch_seqlen,
#                                                                   Y_placeholder: test_target_feed,
#                                                                   keep_prob_: 1.0
#                                                                   })
#                 bagAccu_merge.append(bagAccu)
#                 Y_pred_merge.append(Y_pred)
#                 inst_pred_merge.append(inst_pred)
# =============================================================================


            testIdx = FLAGS.dataset.testIdx
            random_sequences = np.reshape(test_multi_X,(-1,test_multi_X.shape[2],test_multi_X.shape[3]))
            batch_seqlen = np.reshape(seqLenMatrix[testIdx,:],(-1))
            test_target_feed = test_multi_Y.astype(np.int32) 
            bagAccu, Y_pred, inst_pred = sess.run([accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],
                                                   feed_dict={FLAGS.lstm.p_input:random_sequences,
                                                              FLAGS.lstm.seqlen: batch_seqlen,
                                                              Y_placeholder: test_target_feed,
                                                              keep_prob_: 1.0
                                                              })

            print('Epochs %d: accuracy = %.5f '  % (epochs+1, bagAccu)) 
            text_file.write('Epochs %d: accuracy = %.5f\n\n'  % (epochs+1, bagAccu))
            
#            result = sess.run(merged,feed_dict={input_pl: test_input_feed,
#                                                Y_placeholder: test_target_feed,
#                                                keep_prob_: 1.0
#                                                })
            #i = epochs * num_train/FLAGS.finetune_batch_size +step
#            summary_writer.add_summary(result,epochs)
            #baseline = 1-len(test_Y.nonzero())/len(test_Y)
            #if bagAccu > baseline:
             
            # image logit
# =============================================================================
#             if (epochs+1) % 40 == 0:
#                 train_logits_str = GenerateSummaryStr('train_tactic_logits{0}'.format(epochs+1),tf.summary.image,
#                                         Y_logits_image,batch_multi_X,batch_multi_Y,sess,input_pl,Y_placeholder,keep_prob_)
#                              
#                 summary_writer.add_summary(train_logits_str)
#                 test_logits_str = GenerateSummaryStr('test_tactic_logits{0}'.format(epochs+1),tf.summary.image,
#                                         Y_logits_image,test_multi_X,test_multi_Y,sess,input_pl,Y_placeholder,keep_prob_)
#                              
#                 summary_writer.add_summary(test_logits_str)           
# =============================================================================
            
            
            bagAccu,pAccu = calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,FLAGS)
            text_file.write('bag accuracy %.5f, inst accuracy %.5f\n' %(bagAccu, pAccu))
            
            filename = FLAGS._confusion_dir + '/Fold{0}_Epoch{1}_test.csv'.format(fold,epochs)
            ConfusionMatrix(Y_pred,test_multi_Y,FLAGS,filename)
            #print("")

                    #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    #summary_writer.add_summary(summary_str, step)
                    #summary_img_str = sess.run(
                    #    tf.image_summary("training_images",
                    #                tf.reshape(input_pl,
                    #                        (FLAGS.batch_size,
                    #                         FLAGS.image_size,
                    #                         FLAGS.image_size, 1)),
                    #             max_images=FLAGS.batch_size),
                    #    feed_dict=feed_dict
                    #)
                    #summary_writer.add_summary(summary_img_str)
                    
#            for b in range(instNet.num_hidden_layers + 1):
#                if sess.run(tf.is_variable_initialized(instNet._b(b+1))):
#                    #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                    text_file.write("%s with value after fine-tuning\n %s\n" % (instNet._b(b+1).name, instNet._b(b+1).eval(sess)))
#            text_file.close()
#            save_path = saver.save(sess, model_ckpt)
#            print("Model saved in file: %s" % save_path)    
#        else:
#            saver = tf.train.import_meta_graph(model_ckpt+'.meta')
#            saver.restore(sess, model_ckpt)                    
        

           
        testIdx = FLAGS.dataset.testIdx
        random_sequences = np.reshape(test_multi_X,(-1,test_multi_X.shape[2],test_multi_X.shape[3]))
        batch_seqlen = np.reshape(seqLenMatrix[testIdx,:],(-1))
        target_feed = test_multi_Y.astype(np.int32) 
        bagAccu, Y_pred, inst_pred = sess.run([accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],
                                               feed_dict={FLAGS.lstm.p_input:random_sequences,
                                                          FLAGS.lstm.seqlen: batch_seqlen,
                                                          Y_placeholder: test_target_feed,
                                                          keep_prob_: 1.0
                                                          })
        Y_scaled, Y_unscaled = sess.run([tf.nn.softmax(Y),Y],
                                               feed_dict={FLAGS.lstm.p_input:random_sequences,
                                                          FLAGS.lstm.seqlen: batch_seqlen,
                                                          Y_placeholder: test_target_feed,
                                                          keep_prob_: 1.0
                                                          })

        inst_pred_matrix = np.empty([test_target_feed.shape[0],max(num_inst),test_target_feed.shape[1]])
        inst_pred_matrix.fill(np.nan)
        for test_id in range(test_target_feed.shape[0]):
            for k in range(len(FLAGS.C5k_CLASS)):
                for c in range(len(FLAGS.C5k_CLASS[k])):
                    realTacticID = FLAGS.C5k_CLASS[k][c]
                    inst_pred_matrix[test_id,:,realTacticID] = np.exp(inst_pred[k][test_id,:,c])
                    
        test_inst_label = np.empty([test_target_feed.shape[0],max(num_inst)])
        test_inst_label.fill(np.nan)
        for test_id in range(len(test_multi_label)):
            k = np.sum(test_multi_label[test_id,:])
            k_idx = FLAGS.k.index(k)
            inst_gt = FLAGS.playerMap[k_idx].index(test_multi_label[test_id,:].tolist())
            test_inst_label[test_id,inst_gt] = 1.0
            
        
        print('\nAfter %d Epochs: accuracy = %.5f'  % (epochs+1, bagAccu))
        calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,FLAGS)
        time.sleep(0.5)
        
# =============================================================================
#         train_labels_str = GenerateSummaryStr('train_tactic_labels',tf.summary.image,
#                                         Y_labels_image,batch_multi_X,batch_multi_Y,sess,input_pl,Y_placeholder,keep_prob_)
#                              
#         summary_writer.add_summary(train_labels_str)
#         test_labels_str = GenerateSummaryStr('test_tactic_labels',tf.summary.image,
#                                 Y_labels_image,test_multi_X,test_multi_Y,sess,input_pl,Y_placeholder,keep_prob_)
#                      
#         summary_writer.add_summary(test_labels_str)                  
# =============================================================================
        
        
        filename = FLAGS._confusion_dir + '/Fold{0}_Epoch{1}_test_final.csv'.format(fold,epochs)
        ConfusionMatrix(Y_pred,test_multi_Y,FLAGS,filename)        
 
        summary_writer.close()           

def label2image(label_vec):
    try: 
        if len(label_vec.shape) == 2:
            labelImage = tf.expand_dims(tf.expand_dims(label_vec,axis = 2),axis=0)
            labelImage = tf.transpose(labelImage,perm=[0,2,1,3])
            return labelImage
    except:
        print('label dim must be 2!')

def GenerateSummaryStr(tag,summary_op,tf_op,input_data,label_data,sess,input_pl,target_pl,keep_prob):
    input_feed = np.transpose(input_data, (1,0,2))
    target_feed = label_data.astype(np.int32)
    summary_str = sess.run(summary_op(tag,tf_op),
                                    feed_dict={input_pl: input_feed,
                                               target_pl: target_feed,
                                               keep_prob: 1.0
                                               })
    return summary_str

def ConfusionMatrix(logits,labels,FLAGS,filename):
    C = np.zeros((len(FLAGS.tacticName),len(FLAGS.tacticName)))
    CM = C    
    flattenC5k = [val for sublist in FLAGS.C5k_CLASS for val in sublist]
    for bagIdx in range(len(labels)):
        gt = np.argmax(labels[bagIdx])
        #pred = np.argmax(logits[bagIdx])
        pred = logits[bagIdx]
        new_gt = flattenC5k[gt]
        new_pred= flattenC5k[pred]
        C[new_gt,new_pred] = C[new_gt,new_pred] + 1
        
    print(C)
    cumC = np.sum(C,axis=1)
    
    for p in range(len(C)):
        CM[p,:] = np.divide(C[p,:],cumC[p])
    
    df = pd.DataFrame(CM)
    df.round(3)
    df.to_csv(filename)    