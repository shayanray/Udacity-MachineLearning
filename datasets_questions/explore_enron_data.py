#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "count >> ", len(enron_data)

num_poi = 0
num_email = 0
num_salary = 0
num_total_payments_NaN = 0
num_total_payments_NaN_poi = 0

for key, value in enron_data.items():
    num_features = len(value)

    if enron_data[key]['poi'] == 1:
        num_poi +=1

    if enron_data[key]['salary']  != 'NaN':
        num_salary +=1

    if enron_data[key]['email_address'] != 'NaN':
        num_email +=1

    if enron_data[key]['total_payments'] == 'NaN':
        num_total_payments_NaN +=1

    if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] == 1:
        num_total_payments_NaN_poi +=1

print "num_features >> ", num_features
print "num_poi >> ", num_poi
print "num_salary >> ", num_salary
print "num_email >> ", num_email
print "num_total_payments_NaN > ", num_total_payments_NaN
print "%NaN for total payments >> ", (float(num_total_payments_NaN)/len(enron_data)) * 100
print "%NaN for total payments poi >> ", (float(num_total_payments_NaN_poi)/len(enron_data)) * 100
print "stock of james prentice >> ", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "number of emails from Wesley Colwell to POI >> ", enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "number of stock options exercised by Jeffrey K Skilling >> ", enron_data['SKILLING JEFFREY K']['exercised_stock_options']


