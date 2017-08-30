
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:58:32 2017

@author: PeterTsai
"""
import os
import scipy.io
import numpy as np

def read(datadir, fileName):
    #print(os.path.join(os.path.sep,os.getcwd(),datadir,fileName))
    
    #print(os.path.exists(os.path.join(os.path.sep,os.getcwd(),datadir,fileName)))
    mat = scipy.io.loadmat(os.path.join(os.path.sep,os.getcwd(),datadir,fileName))
    
    #print(mat['bags'].shape)
    #print(mat['bags']['instance'].shape)
    #print(mat['bags']['instance'].dtype)
    #print(mat['bags']['instance'][0][0].shape)
    bags = mat['bags']

    #strHello = "the length of (%s) is %d" %('Hello World',len('Hello World'))
    #print(strHello)
    
#    strBagShape = "the shape of bags is ({0},{1})".format(bags.shape[0],bags.shape[1])
#    print(strBagShape)
    #print(bags.shape)
    
    bagNum = len(bags['instance'][0])
    instNum, featDim = (bags['instance'][0][0]).shape
    X = np.zeros([bagNum,instNum,featDim])
    #print(len(bags['instance'][0]))
    #for b in range(len(bags['instance'][0])):
    #    X = bags['instance'][0,b]
    #    print(np.sum(X,axis=1))
    #    Bag1 = np.array(bags['instance'][0][0])
    #    print(np.sum(Bag1,axis=1))
    #X = np.array(bags['instance'])
    for b in range(len(bags['instance'][0])):
        X[b,:,:] = bags['instance'][0,b]

    #print(len(bags['inst_label'][0]))
    Y = np.zeros((len(bags['label'][0]),1))
    for b in range(len(bags['label'][0])):
        Y[b] = int(bags['label'][0][b])
    
    labels = np.zeros([bagNum,instNum])
    for i in range(len(bags['inst_label'][0,:])):
        labels[i,:] = np.squeeze(bags['inst_label'][0][i][0])
        #print(label.shape)
        #print(label.dtype)
        #print(label)
        #print(np.sum(label,axis=1))
    
    #print(Y)
    return X,Y,labels

def multi_class_read(datadir,file_str,num_inst,FLAGS):
    instIdx = np.insert(np.cumsum(num_inst),0,0)
    newIdx = 0
    for k in range(len(FLAGS.k)):
        for c in range(len(FLAGS.C5k_CLASS[k])):
            tactic = FLAGS.tacticName[FLAGS.C5k_CLASS[k][c]]
            fileName= file_str.format(tactic,FLAGS.k[k])
            data_X, data_Y, data_KIlabel = read(datadir,fileName)
            data_KPlabel = Inst2Player(data_KIlabel,FLAGS.playerMap[k])
            if k == 0 and c == 0:
                multi_X = np.zeros([data_X.shape[0],np.sum(num_inst),data_X.shape[2]])
                multi_Y = np.zeros([data_Y.shape[0],len(FLAGS.tacticName)]) 
                multi_KPlabel = np.zeros([len(data_KIlabel),5])
            
            if c == 0:
                multi_X[:,instIdx[k]:instIdx[k+1],:] = data_X
            
            # change original tactic order to multi-mi-Net order
            multi_Y[:,newIdx:newIdx+1] = data_Y
            #KP outside positive bag is 0, so we can use addition to replace concatenate
            multi_KPlabel = multi_KPlabel + data_KPlabel  
            newIdx = newIdx+1
            #multi_KPlabel
    return multi_X, multi_Y, multi_KPlabel

def Inst2Player(KIlabel,playerMap):
    KPlabel = np.zeros([len(KIlabel),5])
    vid, inst = np.nonzero(KIlabel)
    for v in range(len(vid)):
        KPlabel[vid[v],:] = playerMap[inst[v]]
        
    return KPlabel
        
        
    

if __name__ == "__main__":
    datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)'
    fileName = 'EVZoneVelocitySoftAssign(R=16,s=10)3.mat'
    
    X_batch, Y_batch = read(datadir,fileName)
    #print(X_batch[:,1])
    #print(type(X_batch))
    #print(type(Y_batch))
