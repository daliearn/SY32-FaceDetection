# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:34:22 2018

@author: arnaud
"""

import numpy as np
from sklearn import svm
from sklearn.utils import shuffle


def trainClassifier(trainPositives, trainNegatives):
    #concatenation labelization and training of the classifier
    labelPositive = np.ones(len(trainPositives), dtype = bool)
    labelNegative = np.zeros(len(trainNegatives), dtype = bool)
    
    trains = np.concatenate((trainPositives, trainNegatives))
    labels = np.concatenate((labelPositive, labelNegative))
    
    trains, labels = shuffle(trains, labels)
    
    #Training classifier
    clf = svm.SVC(C = 1,kernel= 'rbf')
    clf.fit(trains, labels)
    return clf
