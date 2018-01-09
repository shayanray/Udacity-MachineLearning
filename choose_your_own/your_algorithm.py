#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import tree


neigh = KNeighborsClassifier(n_neighbors = 10, weights='distance')
neigh.fit(features_train, labels_train)
outcome_knn = neigh.predict(features_test)
print " KNN accuracy >> ", accuracy_score(outcome_knn, labels_test) #94%

ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=200)
ada.fit(features_train, labels_train)
outcome_ada = ada.predict(features_test)
print "ADA accuracy >> ", accuracy_score(outcome_ada, labels_test) # 92.4%

rf = RandomForestClassifier(bootstrap=True, max_depth=2, max_features='auto', n_estimators = 200)
rf.fit(features_train, labels_train)
outcome_rf = rf.predict(features_test)
outcome_rf_prb = rf.predict_proba(features_test)
print "RF accuracy >> ", accuracy_score(outcome_rf, labels_test) #
#print "RF probabilities >> ",outcome_rf_prb









try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
