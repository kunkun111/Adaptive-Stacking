# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:05:30 2022

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
import seaborn as sns
import time
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
np.random.seed(0)



# load .arff dataset
#---------------------------------
def load_arff(path, dataset_name, seeds):
    file_path = path + dataset_name + '/'+ dataset_name + str(seeds) + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])


# model
#---------------------------------
def adaptive_stacking_method(data1, data2, ini_train_size, win_size, seeds, name):
    
    data1 = data1.values
    data2 = data2.values
    
    ini_train_size = 100
    win_size = 100
    
    x_train1 = data1[0: 80, :-1]
    y_train1 = data1[0: 80, -1]
    
    x_train2 = data2[0: 80, :-1]
    y_train2 = data2[0: 80, -1]
    
    x_val1 = data1[80: ini_train_size, :-1]
    y_val1 = data1[80: ini_train_size, -1]
    
    x_val2 = data2[80: ini_train_size, :-1]
    y_val2 = data2[80: ini_train_size, -1]
    
    s1_label = np.ones(x_val1.shape[0])
    s2_label = np.full(x_val2.shape[0], 2)
    
    
    
    # fit base model
    #---------------------------------
    s1_model1 = DecisionTreeClassifier()
    s1_model2 = svm.SVC()
    s1_model1.fit(x_train1, y_train1)
    s1_model2.fit(x_train1, y_train1)
    
    s2_model1 = DecisionTreeClassifier()
    s2_model2 = svm.SVC()
    s2_model1.fit(x_train2, y_train2)
    s2_model2.fit(x_train2, y_train2)
    
    
    
    # validate base model
    #---------------------------------
    s1_model1_pre = s1_model1.predict(x_val1)
    s1_model2_pre = s1_model2.predict(x_val1)
    
    
    s2_model1_pre = s2_model1.predict(x_val2)
    s2_model2_pre = s2_model2.predict(x_val2)
    
    
    
    # construct meta train set
    #---------------------------------
    x_train = np.vstack((np.vstack((s1_label, s1_model1_pre, s1_model2_pre)).T, np.vstack((s2_label, s2_model1_pre, s2_model2_pre)).T))
    y_train = np.hstack((y_val1, y_val2))
    
    
    
    # train the meta model
    #---------------------------------
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    
    
    
    # k-fold
    #---------------------------------
    kf = KFold(int((data1.shape[0] - ini_train_size) / win_size))
    stream1 = data1[ini_train_size:, :]
    stream2 = data2[ini_train_size:, :]
    pred = np.zeros(stream1.shape[0])
    
    
    s1_chunk_acc = []
    s2_chunk_acc = []
    
    
    s1_acc_ini = 0
    s2_acc_ini = 0
    
    s2_pred_cum = np.empty(0)
    
    
    # model learning
    for train_index, test_index in tqdm(kf.split(stream1), total = kf.get_n_splits(), desc = "#batch"):
            
        x_test1 = stream1[test_index, :-1]
        y_test1 = stream1[test_index, -1]
        
        x_test2 = stream2[test_index, :-1]
        y_test2 = stream2[test_index, -1]
        
        
        s1_model1_pred = s1_model1.predict(x_test1)
        s1_model2_pred = s1_model2.predict(x_test1)
        
        s2_model1_pred = s2_model1.predict(x_test2)
        s2_model2_pred = s2_model2.predict(x_test2)
        
        s1_label = np.ones(x_test1.shape[0])
        s2_label = np.full(x_test2.shape[0], 2)
        
        
        # construct meta test set
        x_test = np.vstack((np.vstack((s1_label, s1_model1_pred, s1_model2_pred)).T, np.vstack((s2_label, s2_model1_pred, s2_model2_pred)).T))
        y_test = np.hstack((y_test1, y_test2))
        
        
        # test the meta model
        pred = model.predict(x_test)
        
        
        # B1 get the predict results of S1 and S2
        s1_y_pred = pred[:100]
        s2_y_pred = pred[100:]
        s2_pred_cum = np.hstack((s2_pred_cum, s2_y_pred))
        
        
        # get the chunk accuracy
        s1_acc = metrics.accuracy_score(y_test1, s1_y_pred)
        s1_chunk_acc.append(s1_acc)
        
        s2_acc = metrics.accuracy_score(y_test2, s2_y_pred)
        s2_chunk_acc.append(s2_acc)
        
        '''
        # B2 retrain the base model
        if s1_acc < s1_acc_ini:
            s1_model1 = DecisionTreeClassifier()
            s1_model2 = svm.SVC()
            s1_model1.fit(x_test1, y_test1)
            s1_model2.fit(x_test1, y_test1)
        
    
        if s2_acc < s2_acc_ini:
            s2_model1 = DecisionTreeClassifier()
            s2_model2 = svm.SVC()
            s2_model1.fit(x_test2, y_test2)
            s2_model2.fit(x_test2, y_test2)
        
        
        # B3 update the meta test set
        if s1_acc < s1_acc_ini and s2_acc >= s2_acc_ini:
            x_test_new = np.vstack((np.vstack((s1_label, s1_model1_pred, s1_model2_pred)).T, x_train[100:], np.vstack((s2_label, s2_model1_pred, s2_model2_pred)).T))
            y_test_new = np.hstack((y_test1, y_train[100:], y_test2))
            model = GradientBoostingClassifier()
            model.fit(x_test_new, y_test_new)
            
        elif s1_acc >= s1_acc_ini and s2_acc < s2_acc_ini:
            x_test_new = np.vstack((x_train[:100], np.vstack((s1_label, s1_model1_pred, s1_model2_pred)).T,  np.vstack((s2_label, s2_model1_pred, s2_model2_pred)).T))
            y_test_new = np.hstack((y_train[:100], y_test1, y_test2))
            model = GradientBoostingClassifier()
            model.fit(x_test_new, y_test_new)
            
        elif s1_acc >= s1_acc_ini and s2_acc >= s2_acc_ini:
            x_test_new = np.vstack((x_train[:100], np.vstack((s1_label, s1_model1_pred, s1_model2_pred)).T,  x_train[100:], np.vstack((s2_label, s2_model1_pred, s2_model2_pred)).T))
            y_test_new = np.hstack((y_train[:100], y_test1, y_train[100:], y_test2))    
            model = GradientBoostingClassifier()
            model.fit(x_test_new, y_test_new)
       '''     
        
        s1_acc_ini = s1_acc
        s2_acc_ini = s2_acc
        
        x_train = x_test
        y_train = y_test
        
    
    
    # # get the average accuracy
    # s1_ave_acc = np.mean(s1_chunk_acc)
    # s2_ave_acc = np.mean(s2_chunk_acc)
    
    # print('s1 accuracy:', s1_ave_acc)
    # print('s2 accuracy:', s2_ave_acc)
    
    
    # evaluate S2 stream
    Y = data2[ini_train_size:,-1]
    acc = metrics.accuracy_score(Y, s2_pred_cum)
    f1 = metrics.f1_score(Y, s2_pred_cum, average='macro')
    
    print("acc:", acc)
    print("f1:", f1)
    
    # save results
    result = np.zeros([Y.shape[0], 2])
    result[:, 0] = s2_pred_cum
    result[:, 1] = Y
    np.savetxt(str(name) + str(seeds) + 'results.out', result, delimiter=',') 
    
    return acc, f1




if __name__ == '__main__':
    
    
    path = '.../synthetic data/'
    datasets = ['SEAa', 'RTG', 'RBF', 'RBFr', 'AGRa', 'HYP']

    
    for i in range (1, len(datasets)):
        
        acc_total = []
        f1_total = []
        time_total = []
    
        for j in range(15):
            
            data1 = load_arff(path, datasets[0], j)
            data2 = load_arff(path, datasets[0 + i], j)
            
            print(datasets[0], j, datasets[0 + i], j)
            ACC, F1 = adaptive_stacking_method(data1, data2, ini_train_size = 100, win_size = 100, seeds = j, name = datasets[0 + i])
            
            acc_total.append(ACC)
            f1_total.append(F1)
        
        
        print('-----------------------------------------')
        print('AVE Accuracy:', np.mean(acc_total))
        print('STD Accuracy:', np.std(acc_total))
        print('-----------------------------------------')
        print('AVE F1:', np.mean(f1_total))
        print('STD F1:', np.std(f1_total))
        print('-----------------------------------------')

