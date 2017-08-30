#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:58:13 2017

@author: PeterTsai
"""
def base(optimizer='Adam'):
    # load base setting, mainly folder name
    FLAGS = lambda: None
    FLAGS.exp_dir = 'experiment/'+optimizer
    

def normal():
    #FLAGS
    FLAGS = base()
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
    
    return FLAGS


def develop():
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
    FLAGS.finetuning_epochs_epochs = 10
    
    FLAGS.pre_layer_learning_rate = []   
    FLAGS.keep_prob = 1.0
    
    return FLAGS