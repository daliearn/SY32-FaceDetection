# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:08:42 2018

@author: arnaud
"""

import numpy as np
from skimage.io import imread, imshow
from matplotlib.pyplot import show
from skimage.transform import resize
from skimage.filters import scharr

from skimage.feature import hog

from slidingWindow import slidingWindow
from whichBox import groupFaces

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from skimage.feature import hog
import config


def testOneImage(clf, imgFile = "projetface/test/0001.jpg"):
  
    slidingWindowSize = config.slidingWindowSize
    scaleFactor = config.scaleFactor
    accelerator = config.accelerator
    b = 0
    img = imread(imgFile, as_grey= "TRUE")
    
    black_img = np.zeros((max(len(img), len(img[0])), max(len(img), len(img[0]))))        
    black_img[0: len(img), 0: len(img[0])] = img     
    img = black_img  
    
        
    
    
    facesDetected = np.array([])

    h = len(img) / accelerator
    l = len(img[0]) / accelerator 
    
    img = resize(img, (int(h), int(l)))
 
    trueSizeFactor = 1    

    while(h > slidingWindowSize and l > slidingWindowSize):
        print(h)
        print(l)
        AllBoxes = slidingWindow(h, l)
        print(len(AllBoxes))
    
        for i in AllBoxes:
            b += 1
            box = img[
                int(i[1]): int(i[1])+int(i[2]),
                int(i[0]): int(i[0])+int(i[2])
                ]
            #Magic trick to deal with side collisions of my sliding window    
            blackBox = np.zeros((slidingWindowSize,slidingWindowSize))
            blackBox[0: len(box), 0:len(box[0])] = box
              
            box = blackBox              
              
            fd = hog(box, pixels_per_cell=config.Cell, orientations=9)
            
            if (clf.predict(fd.reshape(1, -1))):
                
                #allow to get true boxes and not only 32 pixel boxes with fake dims
                for j in range(len(i)):
                    i[j] = i[j] * accelerator * (1/scaleFactor)**(trueSizeFactor - 1) 
                                
                facesDetected = np.append(facesDetected, i)
                facesDetected = np.append(facesDetected, clf.decision_function(fd.reshape(1, -1))[0])
                
        h = h * scaleFactor
        l = l * scaleFactor
        img = resize(img, (int(h), int(l)))
        trueSizeFactor += 1
            
    #print(len(facesDetected))
    facesDetected = np.array(facesDetected)
    facesDetected = np.reshape(facesDetected, (len(facesDetected) / 4, 4))    

    facesDetected2 = groupFaces(facesDetected)

    im = np.array(Image.open(imgFile), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    
    facesDetected3 = max(facesDetected2, key = lambda x: x[3])
    
    for i in range(len(facesDetected2)):
        x = facesDetected2[i][0]
        y = facesDetected2[i][1]
        d = facesDetected2[i][2]
        rect = patches.Rectangle((x,y),d,d,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
    x = facesDetected3[0]
    y = facesDetected3[1]
    d = facesDetected3[2]
    rect = patches.Rectangle((x,y),d,d,linewidth=3,edgecolor='g',facecolor='none')
    ax.add_patch(rect)
    show()


    print(len(facesDetected2), "/", b)
    return facesDetected2
    