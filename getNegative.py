# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:28:19 2018

@author: arnaud
"""
import numpy as np
import random
from skimage.io import imread, imshow
from matplotlib.pyplot import show
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.filters import gaussian


def getNegative(trainFiles, negativeFactor, imageLength, label):
    trainNegatives = np.zeros((negativeFactor * len(trainFiles), imageLength*imageLength))
    
    #generate negatives randomly
    
    #TODO do something to not take the labelized areas for negatives
    index = 0
    
    for i in range(len(trainFiles)) :
        trainImg = np.array(imread(trainFiles[i], as_grey = "TRUE"))
        trainImg = img_as_float(trainImg)
        
        #Bluring the face zone to not train our classifier in a bad way
        x = label[i, 1]
        y = label[i, 2]
        l = label[i, 3]
        h = label[i, 4]
                
        trainImg[y:y+h,x:x+l] = gaussian(trainImg[y:y+h,x:x+l], sigma = 10)        
        
        for j in range(negativeFactor) :
            xpos = random.randrange(1, len(trainImg) - 11)
            ypos = random.randrange(1, len(trainImg[0]) - 11)
            xwide = random.randrange(10, len(trainImg) - xpos - 1)
            ywide = random.randrange(10, len(trainImg[0]) - ypos - 1)
            
            #TODO Better do a square         
            negImage = trainImg[xpos:xpos+xwide, ypos:ypos+ywide]
            negImage = resize(negImage, (imageLength,imageLength))
            if (index < 3) :
                imshow(negImage)
                show()
            negImage = np.reshape(negImage, imageLength*imageLength)
            trainNegatives[index] = negImage
            index += 1
            
    print("Negative Images generated")        
    return trainNegatives