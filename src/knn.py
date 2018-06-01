__author__ = 'Vikram'
import numpy as np
from sklearn import preprocessing, model_selection, neighbors,metrics
import pandas as pd
import time

start_time = time.time()

#Uncomment this and comment the lines after that for a web-csv read
dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])


'''dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])

'''

print("Getting file from the internet, please wait")
print()
#Drop missing data values
dataFrame.replace('?',np.nan, inplace=True)
dataFrame.dropna(0,'any',inplace=True)
#Drop the ID column
dataFrame.drop(['id'],1,inplace=True)


#Separate attributes and class
X = np.array(dataFrame.drop('class',1))
y = np.array(dataFrame['class'])

kfold = model_selection.KFold(n_splits = 10);
globalAccuracy = 0.0

knn = neighbors.KNeighborsClassifier(n_neighbors = 5)

#Cross-Validate
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(train_index)
    print("--------")

    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    metrics.confusion_matrix(y_test, knn.predict(X_test))
    print(metrics.classification_report(y_test, knn.predict(X_test)))
    globalAccuracy += accuracy
    print(accuracy)

print()
print()
print("--------Global-------")
print(globalAccuracy/10)

print(time.time() - start_time)