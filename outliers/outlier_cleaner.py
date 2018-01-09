#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    print "len(predictions) >> ",len(predictions)
    print "len(ages) >> ", len(ages)
    print "len(net_worths) >> ", len(net_worths)
    #http: // napitupulu - jon.appspot.com / posts / outliers - ud120.html # sample
    for prediction,age,net_worth in zip(predictions,ages,net_worths):
        error = pow(prediction - net_worth, 2)
        #print "age, net_worth, error >> ", age, net_worth, error
        cleaned_data.append((age, net_worth, error),)

    cleaned_data = sorted(cleaned_data, key=lambda x:x[2][0], reverse=True)
    limit = int(len(ages) * 0.1)

    print "limit >> ",limit
    return cleaned_data[limit:]

