# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:51:10 2018

@author: Larry
"""
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
    
    Code borrowed from the following blog:
    https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    '''
    
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
     val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
     val_targ = self.model.validation_data[1]
     _val_f1 = f1_score(val_targ, val_predict)
     _val_recall = recall_score(val_targ, val_predict)
     _val_precision = precision_score(val_targ, val_predict)
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     print("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
     return