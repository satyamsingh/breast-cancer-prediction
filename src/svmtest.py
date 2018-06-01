import numpy as np
from sklearn import preprocessing, model_selection, tree, metrics
import pandas as pd
from sklearn import svm
import time



print("SVM on Wisconsin Breast Cancer Data")
print("Getting file from the internet, please wait")
print()
#Uncomment this and comment the lines after that for a web-csv read
dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])


'''dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])

'''

print("Done reading and parsing file")
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
kfold = model_selection.KFold(n_splits = 10);
globalAccuracy = 0.0

print()

svm = svm.SVC(kernel='poly',degree=3)
i = 0
for train_index, test_index in kfold.split(X):
    i = i+1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(train_index)
    print("---Iteration ",i,"---")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_train)
    #Accuracy
    accuracy = svm.score(X_test, y_test)
    globalAccuracy+=accuracy
    print(accuracy)
    print()
    print()


print("Overall Accuracy: ",globalAccuracy/10)

print("Time Taken: ", time.time() - start_time)