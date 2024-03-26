# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score,precision_recall_curve
import sklearn.metrics as m
from utils import write_log
from config5 import TRAIN_HISTORY

#添加指标：ACC, AUPR, AUC-ROC, F1 +std

class KGCNMetric(Callback):
    def __init__(self, x_train, y_train, x_valid, y_valid,aggregator_type,dataset,K_fold):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.aggregator_type=aggregator_type
        self.dataset=dataset
        self.k=K_fold
        self.threshold=0.5
        # self.user_list, self.train_record, self.valid_record, \
        #     self.item_set, self.k_list = self.topk_settings()

        super(KGCNMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_valid).flatten()
        y_true = self.y_valid.flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)# roc曲线的auc
        precision, recall, _thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr=m.auc(recall,precision)
        y_pred = [1 if prob >= self.threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        
        # print(type(aupr))
        # print(type(auc))
        # print(type(acc))
        # print(type(f1))
        '''
        打印错误数据的数据类型type()，发现数据的类型为numpy.float64，python的内置类型float可以写入json，
        然而numpy类型的float不能写入json，所以应将numpy.float64转换成python内置的数据类型float
        附：numpy.array同样不能写入json文件，需要将numpy.array转换成list，例如a.tolist()
        '''
        logs['val_aupr'] = float(aupr)
        logs['val_auc'] = float(auc)
        logs['val_acc'] = float(acc)
        logs['val_f1'] = float(f1)
        
        logs['dataset']=self.dataset
        logs['aggregator_type']=self.aggregator_type
        logs['kfold']=self.k
        logs['epoch_count']=epoch+1
        print(f'Logging Info - epoch: {epoch+1}, val_auc: {auc}, val_aupr: {aupr}, val_acc: {acc}, val_f1: {f1}')
        write_log('log/'+TRAIN_HISTORY[self.dataset], logs, mode='a')

    @staticmethod
    def get_user_record(data, is_train):
        user_history_dict = defaultdict(set)
        for interaction in data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label == 1:
                user_history_dict[user].add(item)
        return user_history_dict

