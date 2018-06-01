import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import seaborn as sns
import time

print("Random Forest classifier on Wisconsin Breast Cancer Data")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier

sns.set(style='whitegrid')
feat_labels = dataFrame.columns
rf = RandomForestClassifier(n_estimators = 100, random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test,y_test)
print("Accuracy with Random Forest: ",accuracy)
print("Time Taken: ",time.time()-start_time)


#Plot feature importance using Random Forest
importances = rf.feature_importances_
indicies = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indicies],
        color='skyblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()