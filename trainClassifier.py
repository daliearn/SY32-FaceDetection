# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:34:22 2018

@author: arnaud
"""

import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from skimage.io import imread, imshow, imshow_collection
from matplotlib.pyplot import show
from sklearn.ensemble import AdaBoostClassifier as ABC

import config

def trainClassifier(trainPositives, trainNegatives):
    #concatenation labelization and training of the classifier
    labelPositive = np.ones(len(trainPositives), dtype = bool)
    labelNegative = np.zeros(len(trainNegatives), dtype = bool)
    
    trains = np.concatenate((trainPositives, trainNegatives))
    labels = np.concatenate((labelPositive, labelNegative))
    
    trains, labels = shuffle(trains, labels)


            

    #Training classifier
    clf = svm.SVC(C = config.C,kernel= config.kernel, class_weight='balanced')
    #clf = ABC(learning_rate=config.learningRate)
    print("SVM generated")    
    clf.fit(trains, labels)
    print("Classifier trained")
    return clf
