
import tensorflow as tf
#from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.contrib.rnn import LSTMCell

import numpy as np
import functools

"""
Future : Modularization
"""

class LSTMAutoencoder(object):
  """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)

  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

  def __init__(self, hidden_num, inputs, seqlen,
    cell=None, optimizer=None, reverse=True, 
    decode_without_input=False):
    """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size 
              (batch_num x elem_num)
      cell : an rnn cell object (the default option 
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """

    self.max_step_num = int(inputs.shape[0]) 
    #self.batch_num = inputs[0].get_shape().as_list()[0]
    self.elem_num = inputs[0].get_shape().as_list()[1]   
    self.input_ = inputs
    
    if cell is None:
      self._enc_cell = LSTMCell(hidden_num)
      self._dec_cell = LSTMCell(hidden_num)
    else :
      self._enc_cell = cell
      self._dec_cell = cell

    with tf.variable_scope('encoder'):
      self.z_codes, self.enc_state = tf.nn.dynamic_rnn(
        self._enc_cell, inputs, dtype=tf.float32, time_major=True,
        sequence_length=seqlen) 
        
    with tf.variable_scope('decoder') as vs:
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32),
        name="dec_weight")
      dec_bias_ = tf.Variable(
        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
        name="dec_bias")

      if decode_without_input:
        #dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32) 
        #              for _ in range(inputs.shape[0])]
        #              #for _ in range(len(inputs))]
        #dec_inputs = tf.zeros(inputs.shape)
        dec_inputs = tf.zeros_like(inputs)
        dec_outputs, dec_state = tf.nn.dynamic_rnn(
          self._dec_cell, dec_inputs,sequence_length=seqlen,
          initial_state=self.enc_state, dtype=tf.float32, time_major=True)
        """the shape of each tensor
          dec_output_ : (step_num x hidden_num)
          dec_weight_ : (hidden_num x elem_num)
          dec_bias_ : (elem_num)
          output_ : (step_num x elem_num)
          input_ : (step_num x elem_num)
        """
        if reverse:
          #dec_outputs = dec_outputs[::-1]
          dec_outputs = tf.reverse_sequence(dec_outputs,seqlen,seq_axis=0,batch_axis=1)
        dec_output_ = tf.reshape(dec_outputs, [-1, hidden_num])
        #dec_output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])
        #dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num,1,1])
        #self.output_ = tf.batch_matmul(dec_output_, dec_weight_) + dec_bias_
        self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
        #self.output_ = tf.reshape(self.output_,inputs.shape)
        self.output_ = tf.reshape(self.output_,[self.max_step_num ,-1,self.elem_num])
      else : 
        dec_state = self.enc_state
        dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_outputs = []
        #for step in range(inputs.shape[0]):
        for step in range(seqlen):
          if step>0: vs.reuse_variables()
          dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
          dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
          dec_outputs.append(dec_input_)
        if reverse:
          #dec_outputs = dec_outputs[::-1]
          dec_outputs = tf.reverse_sequence(dec_outputs,seqlen,seq_axis=0,batch_axis=1)
        #self.output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])
        self.output_ = tf.stack(dec_outputs)

    #self.input_ = tf.transpose(tf.pack(inputs), [1,0,2])
    
    self.last_output = self._last_relevant(self.z_codes, seqlen)
    #self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
    self.loss = tf.reduce_mean(tf.norm(tf.square(self.input_ - self.output_),axis=2))
    self.output = tf.transpose(self.output_, perm=[1,0,2])
    if optimizer is None :
      self.train = tf.train.AdamOptimizer().minimize(self.loss)
      #self.train = tf.train.RMSPropOptimizer(0.5).minimize(self.loss)
    else :
        self.train = optimizer.minimize(self.loss)
  
  @staticmethod
  def _last_relevant(output, length):
      max_length = int(output.get_shape()[0])
      batch_size = tf.shape(output)[1]
      output_size = int(output.get_shape()[2])
      index = tf.range(0, batch_size) * max_length + (length - 1)
      flat = tf.reshape(output, [-1, output_size])
      relevant = tf.gather(flat, index)
      return relevant
  
#  @property
#  def length(self):
#      used = tf.sign(tf.reduce_max(tf.abs(self.input_), reduction_indices=2))
#      length = tf.reduce_sum(used, reduction_indices=1)
#      length = tf.cast(length, tf.int32)
#      return length      
