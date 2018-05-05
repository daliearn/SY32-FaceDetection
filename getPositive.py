# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:20:53 2018

@author: arnaud
"""

import numpy as np
from skimage.io import imread, imshow
from matplotlib.pyplot import show
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.feature import hog

import config

def getPositive(trainFiles, label) :

    slidingWindowSize = config.slidingWindowSize
    trainPositives = np.array([])
    
    #generate positives
    for i in range(len(trainFiles)) :
        print(i)
        print("positive")
        trainImg = np.array(imread(trainFiles[i], as_grey = "TRUE"))
        trainImg = img_as_float(trainImg)
        
        x = label[i, 1]
        y = label[i, 2]
        l = label[i, 3]
        h = label[i, 4]
        '''  
        #Goal : get a minimum square with a face in
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
        '''
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
            
        '''
        positive = trainImg[
            (YSmall + YLarge) / 2: (YSmall + YLarge + dsmall + dLarge) / 2,
            (XSmall + XLarge) / 2: (XSmall + XLarge + dsmall + dLarge) / 2
            ]
        '''           
        #We'll work with shapes of image
        #positive = scharr(positive)
        positive = resize(positive,(slidingWindowSize, slidingWindowSize))
        fd = hog(positive, pixels_per_cell=config.Cell, orientations=9)        
        trainPositives = np.append(trainPositives, fd)
        
        positive = np.fliplr(positive)
        fd = hog(positive, pixels_per_cell=config.Cell, orientations=9)        
        trainPositives = np.append(trainPositives, fd)
        
        
    print("Loaded positives")
    trainPositives = np.reshape(trainPositives, (len(trainPositives) / len(fd), len(fd)))
    return trainPositives