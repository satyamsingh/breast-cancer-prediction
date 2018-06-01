import numpy as np
from sklearn import preprocessing, model_selection, tree, metrics, neighbors,svm
import pandas as pd
from sklearn import svm
import time

print("Ensemble Classifier on Wisconsin Breast Cancer Data")
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

#Set up 10 fold cross-validation
kf = model_selection.KFold(n_splits = 10);
globalAccuracy = 0.0



svm = svm.SVC(kernel='poly',degree=3)
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
dt = tree.DecisionTreeClassifier(criterion="gini",max_depth=30,min_samples_leaf=20)



ind = 0

for train_index, test_index in kf.split(X):
    ind = ind+1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(train_index)
    print("----Iteration ",ind,"----")
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    ypred=[]
    for i in range(0,len(X_test)):
        y1 = svm.predict(X_test[i].reshape(-1,9))  #Take a vote from each classifier
        y2 = knn.predict(X_test[i].reshape(-1,9))
        y3 = dt.predict(X_test[i].reshape(-1,9))
     #  print("Predicted Class: ")
     #  print("SVM: ",y1," KNN: ",y2, "Decision Tree: ",y3)
        if(y1[0]+y2[0]+y3[0]>8):
            ypred.append(4)
        else:
            ypred.append(2)

    print("Iteration Confusion Matrix: ")
    metrics.confusion_matrix(y_test, ypred)
    print(metrics.classification_report(y_test, ypred))
    print()


print()
print()
print("Overall Accuracy: ",globalAccuracy/10)
print("Print Time Taken: ",time.time() - start_time)