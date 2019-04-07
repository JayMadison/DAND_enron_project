#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus','total_stock_value','exercised_stock_options',
                 'pct_to_poi'] 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
# Reomve the Total Line
data_dict.pop('TOTAL',None)
print "Data Removed"
### Task 3: Create new feature(s)
for person in data_dict:
    fr = data_dict[person]['from_messages']
    to_poi = data_dict[person]['from_this_person_to_poi']
    fr_poi = data_dict[person]['from_poi_to_this_person']
    to= data_dict[person]['to_messages']
    # Email to/from poi %
    if fr == 'NaN' or to_poi == 'NaN' or fr == 0:
        data_dict[person]['pct_to_poi'] = 0
    else:
        data_dict[person]['pct_to_poi'] = float(fr_poi)/float(to)
    # Email From poi/shared w/ poi %
    if to == 'NaN' or fr_poi == 'NaN' or to == 0:
         data_dict[person]['pct_from_poi'] = 0
    else:
         data_dict[person]['pct_from_poi'] = float(to_poi)/float(fr)
print "Features Done"
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#This one worked really well, but can't really be tuned
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#extremely Slow
#from sklearn.svm import SVC
#clf = SVC(kernel="linear")

# This worked about the best, worth tuning
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()

#Slower and didn't yeild a better result 
#from sklearn.ensemble import AdaBoostClassifier
#clf =  AdaBoostClassifier(n_estimators=100)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#from sklearn.feature_selection import SelectKBest,f_classif
#kbest = SelectKBest(f_classif).fit(features,labels)
#print kbest.scores_
#print kbest.pv


#tree = DecisionTreeClassifier()
#parameters = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,20,50,100,150],'min_samples_split':[2,3,4,5,6]}
#clf = GridSearchCV(tree, parameters)

#Results {'min_samples_split': 2, 'criterion': 'entropy', 'max_depth': 10

clf = DecisionTreeClassifier(min_samples_split=2,criterion='entropy',max_depth= 10)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)