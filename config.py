# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:43:04 2018

@author: arnaud
"""


classifier = "ADB"

C = 10000000 



kernel = 'rbf'

stepFactor = 0.1

learningRate = 1

#Size of the sliding window
slidingWindowSize = 36
scaleFactor = 0.75

accelerator = 3

#Pixel we want in an image 
imageLength = 450

#number of negatives generated for one positive (integer)
negativeFactor = 5

#Cell for hog
Cell = (4, 4)

#ThreshHold for fake positive
AreaTresh = 0.25

#percentage of image used for second step
trainingFactor = 0.1 