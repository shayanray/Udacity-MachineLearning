#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
#print "data_Dict ", data_dict
features = ["salary", "bonus"]
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features)


### your code below
sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
print "sorted data", sorted_data

for point in data:
    salary = point[0]
    bonus = point[1]
    #print " salary, bonus >> ",salary, bonus
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()