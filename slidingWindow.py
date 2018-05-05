# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:15:49 2018

@author: arnaud
"""

import numpy as np
import config

def slidingWindow(h, l):
    
    slidingWindowSize = config.slidingWindowSize
    stepFactor = config.stepFactor
    AllBoxes = np.array([]) 

    #firstWindow is the full image
    x = 0
    y = 0    
        
    
    #Would be better with a do while
    while (x + slidingWindowSize < l - 1 or y + slidingWindowSize < h - 1):
                
        box = [x, y, slidingWindowSize]
        AllBoxes = np.append(AllBoxes, box)        
        
        #Computation of new coordinates    

        #if we are at the end of a row, we go to the next row    
        if (x + slidingWindowSize > l - 1) :
            x = 0
            y = y + slidingWindowSize * stepFactor
        #In all other cases, we go to the next box
        else :
            x = x + slidingWindowSize * stepFactor
        
    AllBoxes = np.reshape(AllBoxes, (len(AllBoxes) / 3, 3))    
    return np.array(AllBoxes)

