#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
t0 = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
timeToTrain = time()-t0
print "Time taken to fit ",round(timeToTrain , 3) , "seconds"

predictStartTime = time()
outcome = clf.predict(features_test)
timetoPredict = time() - predictStartTime
print "Time taken to predict ",round(timetoPredict, 3) , "seconds"


print " Accuracy of predicted outcome >> ", accuracy_score(outcome, labels_test)


#########################################################


