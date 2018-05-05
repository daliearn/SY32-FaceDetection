# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:46:11 2018

@author: arnaud
"""

import glob
import numpy as np

from skimage.io import imread, imshow, imshow_collection
from matplotlib.pyplot import show
from skimage.util import img_as_float, crop
from sklearn import svm
from sklearn.utils import shuffle
from skimage.transform import resize

from whichBox import groupFaces, whichBoxToRemove
from getPositive import getPositive
from getNegative import getNegative
from trainClassifier import trainClassifier
from testOneImage import testOneImage
from falsePosToNeg import falsePosToNeg

from warnings import catch_warnings
from sklearn.externals import joblib

import config 

catch_warnings()

#loading files of the project
trainFiles = glob.glob("projetface/train/*")
trainFiles = np.sort(trainFiles)
label = np.loadtxt("projetface/label.txt", dtype = "int")
testFiles = glob.glob("projetface/test/*")

trainPositives = np.array([])
#generate positives
for i in range(len(trainFiles)) :
    trainImg = np.array(imread(trainFiles[i], as_grey = "TRUE"))
    trainImg = img_as_float(trainImg)
    
    x = label[i, 1]
    y = label[i, 2]
    l = label[i, 3]
    h = label[i, 4]
      
    #Goal : get a minimum square with a face in
    #We assume that this is better 
    if (l < h) :
        d = (h - l) / 2
        YSmall = y+d
        XSmall = x
        positive = trainImg[YSmall: YSmall+l, x: x+l]
        dsmall = l            
    else :
        d = (l - h) / 2
        YSmall = y
        XSmall = x+d
        positive = trainImg[y: y+h, XSmall: XSmall+h]
        dsmall = h
    
    #Goal : get a maximum square with a face in
    if (l < h) :
        d = (h - l) / 2
        XLarge = x - d
        if (XLarge - d < 0):
            XLarge = 0
        YLarge = y
        positive = trainImg[y: y+h, XLarge: XLarge+h]      
        dLarge = h
    else :
        d = (l - h) / 2
        YLarge = y - d
        if (YLarge - d < 0):
            YLarge = 0
        XLarge = x
        positive = trainImg[YLarge : YLarge+l, x: x+l]   
        dLarge = l
        

    positive = trainImg[
        (YSmall + YLarge) / 2: (YSmall + YLarge + dsmall + dLarge) / 2,
        (XSmall + XLarge) / 2: (XSmall + XLarge + dsmall + dLarge) / 2
        ]
        
    positive = np.resize(450,450)
    trainPositives = np.appendpositive

imshow(np.mean(trainPositives))
show()