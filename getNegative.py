# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:28:19 2018

@author: arnaud
"""
import numpy as np
import random
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filters import gaussian

import config

from skimage.feature import hog

def getNegative(trainFiles, label):
    
    negativeFactor = config.negativeFactor
    slidingWindowSize = config.slidingWindowSize
    trainNegatives = np.array([])
    #generate negatives randomly
    #TODO do something to not take the labelized areas for negatives
    for i in range(len(trainFiles)) :
        trainImg = np.array(imread(trainFiles[i], as_grey = "TRUE"))
        trainImg = img_as_float(trainImg)
        
        #Bluring the face zone to not train our classifier on positive examples
        x = label[i, 1]
        y = label[i, 2]
        l = label[i, 3]
        h = label[i, 4]
                
        trainImg[y:y+h,x:x+l] = gaussian(trainImg[y:y+h,x:x+l], sigma = 10)        
        
        #We want to get n negative per positive        
        for j in range(negativeFactor) :
            print(i)
            print(j)
            print("negative")
            #Correct if gaussian blur areas can be considered as FALSE
            ypos = random.randrange(1, len(trainImg) - slidingWindowSize)
            xpos = random.randrange(1, len(trainImg[0]) - slidingWindowSize)
            h = slidingWindowSize                        
            
            #We should better find a way to make these random boxes
            #Not colliding with actual faces
            
            negImage = trainImg[ypos:ypos+h, xpos:xpos+h]
            fd = hog(negImage, pixels_per_cell=config.Cell, orientations=9)
            trainNegatives = np.append(trainNegatives, fd)
            
    print("Negative Images generated")
    trainNegatives = np.reshape(trainNegatives, (len(trainNegatives) / len(fd), len(fd)))        
    return trainNegatives