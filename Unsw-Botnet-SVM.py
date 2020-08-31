# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:45:37 2020

@author: 1804499
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import time
import psutil

#Function to load, normalize and splits datasets in csv format
def Load_Data():
    benign = np.loadtxt("UNSW_NB15_Train.csv", delimiter = ",")
    benscan = np.loadtxt("UNSW_NB15_Test.csv", delimiter = ",")
    alldata = np.concatenate((benign, benscan))
    j = len(benscan[0])
    data = alldata[:, 1:j]
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2)
    return bendata, benmir, benlabel, benslabel

    
# train the classifier

def train_svm(train, trainlabel):
    iotbinclf = SVC(kernel = 'linear')
    trainclf = iotbinclf.fit(train, trainlabel)
    return trainclf

#load data

train, test, trainlabel, testl = Load_Data()


def predicted_svm(clf, test):
    predictions = clf.predict(test)
    acc = metrics.accuracy_score(testl, predictions)
    return acc

def optimz():
    clf = train_svm(train, trainlabel)
    mp = psutil.Process(os.getpid())
    mi = mp.memory_info().rss
    st = time.time()
    acc = predicted_svm(clf, test)
    et = time.time()
    ett = et - st
    mf = psutil.Process(os.getpid())
    mf = mp.memory_info().rss
    mtm = mf - mi
    return mtm, ett, acc

mtm, ett,  ac = optimz()

print(mtm, ett, ac)

    

