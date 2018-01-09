#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from pprint import pprint


def calculateRescaledValues(data_dict):
    salary = []
    ex_stok = []
    for users in data_dict:
        val = data_dict[users]["salary"]
        if val == 'NaN':
            continue
        salary.append(float(val))
        val = data_dict[users]["exercised_stock_options"]
        if val == 'NaN':
            continue
        ex_stok.append(float(val))

    salary = [min(salary), 200000.0, max(salary)]
    ex_stok = [min(ex_stok), 1000000.0, max(ex_stok)]

    print salary
    print ex_stok

    salary = numpy.array([[e] for e in salary])
    ex_stok = numpy.array([[e] for e in ex_stok])

    scaler_salary = MinMaxScaler()
    scaler_stok = MinMaxScaler()

    rescaled_salary = scaler_salary.fit_transform(salary)
    rescaled_stock = scaler_salary.fit_transform(ex_stok)

    print rescaled_salary
    print rescaled_stock

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2] #, feature_3
data = featureFormat(data_dict, features_list )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_fit = scaler.fit_transform(data)

calculateRescaledValues(data_dict)

sorted_data = sorted(data_fit, key=lambda x:x[2], reverse=True)
print " sorted data(2) "
pprint(sorted_data)
print "max exercised_stock_options >> ", max(sorted_data[2])
print "min exercised_stock_options >> ", min(sorted_data[2])

sorted_data = sorted(data_fit, key=lambda x:x[1], reverse=True)
print "\n\n\n\n sorted data(1) "
pprint(sorted_data)
print "max SALARY >> ", max(sorted_data[1])
print "min SALARY >> ", min(sorted_data[1])

poi, finance_features = targetFeatureSplit( data_fit )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features: #,_
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2) #km = KMeans(n_clusters=2) #
km.fit(finance_features)
pred = km.predict(finance_features)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"