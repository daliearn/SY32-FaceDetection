# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:13:13 2018

@author: arnaud
"""

import glob
import numpy as np
import random
from skimage.io import imread, imshow, imshow_collection
from matplotlib.pyplot import show
from skimage.util import img_as_float, crop
from sklearn import svm
from sklearn.utils import shuffle
from skimage.transform import resize

from getPositive import getPositive
from getNegative import getNegative
from trainClassifier import trainClassifier
from testOneImage import testOneImage
from warnings import catch_warnings

catch_warnings()

#loading files of the project
trainFiles = glob.glob("projetface/train/*")
trainFiles = np.sort(trainFiles)
label = np.loadtxt("projetface/label.txt", dtype = "int")
testFiles = glob.glob("projetface/test/*")


#Pixel we want in an image (should be 450 but set to 96 in development phase)
imageLength = 450

#number of negatives generated for one positive (integer)
negativeFactor = 2

###########################################

trainPositives = getPositive(trainFiles, label, imageLength)
trainNegatives = getNegative(trainFiles, negativeFactor, imageLength, label)
        
#TODO data augmentation           
clf = trainClassifier(trainPositives, trainNegatives)

###########################################

for i in range (0,3):
    detectedFaces = testOneImage(clf, testFiles[i])
    img = imread(testFiles[i], as_grey = "TRUE")
    for j in range (0, len(detectedFaces) - 1):
        print(detectedFaces[j])
        xface = int(detectedFaces[j, 0])        
        yface = int(detectedFaces[j, 1])
        hface = int(detectedFaces[j, 2])       

        face = img[int(xface) : int(xface + hface), int(yface) : int(yface + hface)]
        imshow(face)
        show()

