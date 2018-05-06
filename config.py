# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:43:04 2018

@author: arnaud
"""

#SVM settings
C = 10000 
kernel = 'rbf'

#Cell for hog
Cell = (4, 4)

#Sliding window settings
slidingWindowSize = 36
stepFactor = 0.1
scaleFactor = 0.75

#Reduction factor for the image submitted. 
#A large value speed up the algorithm
accelerator = 3.5

#Pixel we want in an image 
imageLength = 450


#Negative samples settings

#ThreshHold for fake positive
AreaTresh = 0.33
#number of negatives generated for one positive (integer)
negativeFactor = 5
#percentage of image used for second step
trainingFactor = 0.1