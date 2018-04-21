# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:20:53 2018

@author: arnaud
"""

import numpy as np
from skimage.io import imread, imshow, imshow_collection
from matplotlib.pyplot import show
from skimage.util import img_as_float
from skimage.transform import resize

def getPositive(trainFiles, label, imageLength) :
    trainPositives = np.zeros((len(trainFiles), imageLength*imageLength))
    #generate positives
    for i in range(len(trainFiles)) :
        trainImg = np.array(imread(trainFiles[i], as_grey = "TRUE"))
        trainImg = img_as_float(trainImg)
        
        x = label[i, 1]
        y = label[i, 2]
        l = label[i, 3]
        h = label[i, 4]
        
        
        positive = trainImg[y:y+h,x:x+l]
        #trainImg[y:y+h,x:x+l] = 0
        #imshow(trainImg)
        #show()
        positive = resize(positive,(imageLength,imageLength))
        positive = np.reshape(positive, imageLength*imageLength)
        trainPositives[i] = positive
        #TODO faire une image noire a 450*450 et pas de resize
        
            
    print("loaded positives")
    return trainPositives