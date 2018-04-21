# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:15:49 2018

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


def slidingWindow(fileName = "projetface/test/0001.jpg", scaleFactor = 0.5, step = 0.25, epsilonPixel = 60, imgLength = 450):
    
    AllBoxes = np.array([])    
    
    img = imread(fileName, as_grey="TRUE")
    img = img_as_float(img)
    img = resize(img ,(imgLength, imgLength))    

    #We assume box is a square so a box is defined by x, y, h
    #firstWindow is the full image
    x = 0
    y = 0    
    h = imgLength - 1    
    
    #Would be better with a do while
    while (h > epsilonPixel):    
        box = [x, y, h]
        AllBoxes = np.append(AllBoxes, box)        
        
        #Computation of new coordinates    

        #if we are at the end of all possible boxes : we reduce the size       
        if (x + h > imgLength - 1 and y + h > imgLength - 1) : 
            h = h * scaleFactor
            x = 0
            y = 0
        #if we are at the end of a row, we go to the next row    
        elif (x + h > imgLength - 1) :
            x = 0
            y = y + h * step
        #In all other cases, we go to the next box
        else :
            x = x + h * step
        
    AllBoxes = np.array(AllBoxes)
    AllBoxes = np.reshape(AllBoxes, (len(AllBoxes) / 3, 3))    
    return np.array(AllBoxes)

