# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:08:42 2018

@author: arnaud
"""

import numpy as np
from skimage.io import imread, imshow
from matplotlib.pyplot import show
from skimage.transform import resize

from slidingWindow import slidingWindow


def testOneImage(clf, imgFile = "projetface/test/0001.jpg", imageLength = 450):
    AllBoxes = slidingWindow()
    #Testing One Image
    img = imread(imgFile, as_grey= "TRUE")
    img = resize(img, (imageLength,imageLength))
    
    facesDetected = np.array([])
    
    for i in AllBoxes:
        print(i)
        box = img[
            int(i[0]): int(i[0]+i[2]),
            int(i[1]): int(i[1]+i[2])
        ]
        '''
        imshow(box)
        show()
        '''
        box = resize(box, (imageLength, imageLength))
        box = np.reshape(box, (imageLength * imageLength))
        if (clf.predict(box)):
            facesDetected = np.append(facesDetected, i)
            print("TRUE")
       
    print(len(facesDetected))
    facesDetected = np.array(facesDetected)
    facesDetected = np.reshape(facesDetected, (len(facesDetected) / 3, 3))
    return facesDetected