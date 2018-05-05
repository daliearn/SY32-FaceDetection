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

###########################################
###########################################
###########################################

learningPhaseIndex = len(trainFiles) - int(len(trainFiles) * config.trainingFactor) 

#Generating positive and Negative examples
trainPositives = getPositive(trainFiles[0:learningPhaseIndex], label[0:learningPhaseIndex])
trainNegatives = getNegative(trainFiles[0:learningPhaseIndex], label[0:learningPhaseIndex])
     
#Step 1 training         
clf = trainClassifier(trainPositives, trainNegatives)

#Step 2 Fake Pos to Neg
trainNegatives = falsePosToNeg(clf, trainFiles[learningPhaseIndex:len(trainFiles)], label[learningPhaseIndex:len(label)], trainNegatives)
clf = trainClassifier(trainPositives, trainNegatives)

joblib.dump(clf, 'classifier.pkl', compress = 9)
'''
'''
