#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train=features_train[:len(features_train)/100]
#labels_train=labels_train[:len(labels_train)/100]

t0 = time()
linear_svm = svm.SVC(kernel="rbf", C=10000) #linear, rbf, poly, sigmoid
linear_svm.fit(features_train, labels_train)
print "time to train ", round(time() - t0, 3), " in seconds"

pred_start_time = time()
pred = linear_svm.predict(features_test)
print "time to predict ", round(time() - pred_start_time, 3), " in seconds"
print " accuracy score is ", accuracy_score(pred, labels_test)

print "prediction[10] ", pred[10]
print "prediction[26] ", pred[26]
print "prediction[50] ", pred[50]

np_arr = np.array(pred)
chris = len(np_arr[np_arr == 1])
print " number of chris predictions out of 1700 ", chris

#########################################################
### your code goes here ###


#########################################################


