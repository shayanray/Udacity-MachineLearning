#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

print features_test
print labels_test

cnt = 0
total = 0
for label in labels_test:
    total +=1
    if label == 1.0:
        cnt +=1

print "poi in test set ", cnt
print "total in test set ", total


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "predicted labels ", pred
print clf.score(features_test, labels_test)

from sklearn.metrics import precision_score, recall_score

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
clf.fit(features,labels)
print "Overfitted Tree ",clf.score(features, labels)
