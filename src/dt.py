import numpy as np
from sklearn import preprocessing, model_selection, tree, metrics
import pandas as pd
import plotly
plotly.tools.set_credentials_file(username = 'WVik', api_key='MyiOZkG5iObapIYvwKVT')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from subprocess import check_call
import time


print("Decision Tree Classifier on Wisconsin Breast Cancer Data")
print("Getting file from the internet, please wait")
print()
#Uncomment this and comment the lines after that for a web-csv read
dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])


'''dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])

'''
print("Done reading and parsing data file")
print()

start_time = time.time()
#Drop missing data values
dataFrame.replace('?',np.nan, inplace=True)
dataFrame.dropna(0,'any',inplace=True)


#Drop the ID column
dataFrame.drop(['id'],1,inplace=True)


#Separate attributes and class
X = np.array(dataFrame.drop('class',1))
y = np.array(dataFrame['class'])

#Create one particular decision tree
X_train_temp,X_test_temp,y_train_temp,y_test_temp = model_selection.train_test_split(X,y,test_size=0.3)


#Create the decision tree as an image file.
clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=15,min_samples_leaf=10).fit(X_train_temp,y_train_temp)
tree.export_graphviz(clf,
    out_file='tree.dot')
check_call(['dot','-Tpng','tree.dot','-o','OutputFile.png'])



#Cross Validation
#Set up 10 fold cross-validation
kf = model_selection.KFold(n_splits = 10);
globalAccuracy = 0.0


i = 0
for train_index, test_index in kf.split(X):
    i = i+1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(train_index)
    print("----Iteration ",i,"----")
    clf_gini = tree.DecisionTreeClassifier(criterion="gini",max_depth=30,min_samples_leaf=20)
    clf_gini.fit(X_train, y_train)
    accuracy = clf_gini.score(X_test, y_test)
    metrics.confusion_matrix(y_test, clf.predict(X_test))
    print(metrics.classification_report(y_test, clf_gini.predict(X_test)))
    print("Iteration accuracy: ",accuracy)
    print()
    print()
    globalAccuracy += accuracy

print()
print()

print("Overall Accuracy: ", globalAccuracy/10)

print("Time taken: " ,time.time() - start_time)