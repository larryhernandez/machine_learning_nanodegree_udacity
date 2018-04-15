# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:51:10 2018

@author: Larry
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback


def microaveraged_f1score(y_true, y_pred):
    '''   '''
    yTrue = [np.argmax(y) for y in y_true]
    yPred = [np.argmax(y) for y in y_pred]
    return f1_score(yTrue, yPred, average = 'micro')


class Metrics(Callback):
    ''' 
    class that calculates f1, precision, and recall scores after 
    
    Code borrowed (and adapted) from the following blog:
    https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    '''
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        return
    
    def on_epoch_end(self, epoch, logs={}):
        val_p = self.model.predict(self.validation_data[0])
        val_predict = [np.argmax(val_p[idx]) for idx in range(len(val_p))]
        val_t = self.validation_data[1]
        val_targ = [np.argmax(val_t[idx]) for idx in range(len(val_t))]
        _val_f1 = f1_score(val_targ, val_predict, average = 'micro')
        val_recall_score = recall_score(val_targ, val_predict, average = 'micro')
        val_precision_score = precision_score(val_targ, val_predict, average = 'micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(val_recall_score)
        self.val_precisions.append(val_precision_score)
        print("— val_f1: {f1}".format(f1 = _val_f1))
        return