# -*- coding: utf-8 -*-
"""
Created on Fri May  4 08:15:04 2018

@author: arnaud
"""

import glob
import numpy as np

from testOneImage import testOneImage

from sklearn.externals import joblib


clf = joblib.load('classifier.pkl')
testFiles = glob.glob("projetface/test/*")
testFiles = np.sort(testFiles)

tests = np.array([])

result = open("result.txt","w") 
cheat = open("cheat.txt","w") 

for i in range(len(testFiles)):
    print(i)
    test = testOneImage(clf, testFiles[i])
    str1 = testFiles[i]
    str1 = str1.replace('projetface/test/','')
    str1 = str1.replace('.jpg','')

    for t in test:
        
        result.write(str1) 
        result.write(" ")
        
        result.write(str(int(t[0])))
        result.write(" ")
        
        result.write(str(int(t[1])))   
        result.write(" ")
        
        result.write(str(int(t[2])))
        result.write(" ")
        
        result.write(str(int(t[2])))
        result.write(" ")
        
        result.write(str(t[3]))
        result.write("\n")
        
    best = max(test, key = lambda x: x[3])
    cheat.write(str1) 
    cheat.write(" ")
    
    cheat.write(str(int(best[0])))
    cheat.write(" ")
    
    cheat.write(str(int(best[1])))   
    cheat.write(" ")
    
    cheat.write(str(int(best[2])))
    cheat.write(" ")
    
    cheat.write(str(int(best[2])))
    cheat.write(" ")
    
    cheat.write(str(best[3]))
    cheat.write("\n")
        