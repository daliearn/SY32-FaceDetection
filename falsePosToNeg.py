# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:04:02 2018

@author: arnaud
"""


import numpy as np

from skimage.io import imread, imshow
from matplotlib.pyplot import show
from skimage.util import img_as_float
from skimage.transform import resize

from whichBox import  whichBoxToRemove
from testOneImage import testOneImage

from skimage.feature import hog

import config

def falsePosToNeg(clf, trainFiles, label, trainNegatives):
    print("Starting generation of false positive")
    slidingWindowSize = config.slidingWindowSize
    
    for i in range(len(trainFiles)):  
        print(i)              
        testedImg = imread(trainFiles[i], as_grey='TRUE')
        testedImg = img_as_float(testedImg)

        x = label[i, 1]
        y = label[i, 2]
        l = label[i, 3]
        h = label[i, 4]
          
        #Goal : get a maximum square with a face in
        if (l < h) :
            d = (h - l) / 2
            X = x - d
            if (X - d < 0):
                X = 0
            positive = testedImg[y: y+h, X: X+h]     
            boxToTest = [X, y, h, 1.0]
        else :
            Y = y - d
            if (Y - d < 0):
                Y = 0
            d = (l - h) / 2
            positive = testedImg[Y : Y+l, x: x+l]            
            boxToTest = [x, Y, l, 1.0]
        
        facesDetected = testOneImage(clf, trainFiles[i])
        
        for face in facesDetected:
                         
            if(whichBoxToRemove(boxToTest, face, config.AreaTresh) == 0):
                neg = testedImg[
                    int(face[1]): int(face[1])+int(face[2]),
                    int(face[0]): int(face[0])+int(face[2])
                    ]
                neg = resize(neg, (slidingWindowSize, slidingWindowSize))
                fd = hog(neg, pixels_per_cell=config.Cell, orientations=9)
                fd = np.reshape(fd, (1, len(fd)))
                trainNegatives = np.concatenate((trainNegatives, np.array(fd)))
    
    
                neg = np.fliplr(neg)
                fd = hog(neg, pixels_per_cell=config.Cell, orientations=9)
                fd = np.reshape(fd, (1, len(fd)))                
                trainNegatives = np.concatenate((trainNegatives, np.array(fd)))                
    print("False Positive added to neg")
    return trainNegatives