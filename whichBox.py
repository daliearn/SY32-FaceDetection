# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 23:46:06 2018

@author: arnaud
"""

import numpy as np
from skimage.io import imread, imshow, imshow_collection
from matplotlib.pyplot import show
from skimage.util import img_as_float, crop
from sklearn import svm
from sklearn.utils import shuffle
from skimage.transform import resize


from warnings import catch_warnings


#return the correct 
def whichBoxToRemove(box1, box2) :
    x1 = box1[0]
    x2 = box2[0]
    y1 = box1[1]
    y2 = box2[1]
    h1 = box1[2]
    h2 = box2[2]
    s1 = box1[3]
    s2 = box2[3]
  
    left = max(x1, x2)
    right = min(x1 + h1, x2 + h2)
    bottom = min(y1 + h1, y2 + h2)
    top = max(y1, y2)
   
    H = bottom - top
    L = right - left
    
    if (H < 0 or L < 0):
        return 0
    
    
    A1 = h1 ** 2    
    A2 = h2 ** 2    
    
    inter = H * L
   
    union = A1 + A2 - inter    
    Area = inter / union
    

    #print(Area)
               
    if (Area > 0.5):
        #print("To delete ")
        if (s1 > s2) :
            return 2
        else :
            return 1
    else :
        return 0



def groupFaces (facesDetected) :
    #facesDetected = sorted(facesDetected, key = lambda x: x[3])

    stopFlag = False
    while (stopFlag == False) :
        actionFlag = False
        i = 0
        while( i < len(facesDetected) - 1):
            j = i + 1
            while(j < len(facesDetected) - 1):
                box = whichBoxToRemove(facesDetected[i], facesDetected[j])
                if (box == 1) :
                    facesDetected = np.delete(facesDetected, i, axis = 0)
                    actionFlag = True
                    i = 0
                elif (box == 2) :
                    facesDetected = np.delete(facesDetected, j, axis = 0)
                    actionFlag = True
                    i = 0
                j += 1
            i += 1
        if (actionFlag == False):            
            stopFlag = True  

    return facesDetected
    
    
    #return np.reshape(ret, (len(ret) / 4, 4))
            