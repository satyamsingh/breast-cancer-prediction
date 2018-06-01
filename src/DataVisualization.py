import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, tree
import pandas as pd


dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])


#Drop missing data values
dataFrame.replace('?',np.nan, inplace=True)
dataFrame.dropna(0,'any',inplace=True)


#Drop the ID column
dataFrame.drop(['id'],1,inplace=True)


#Separate attributes and class
X = np.array(dataFrame.drop('class',1))
y = np.array(dataFrame['class'])

X_1 = X[:,0:1]
X_2 = X[:,1:2]
X_3 = X[:,2:3]
X_5 = X[:,5:6]
#n,bins,patches = plt.hist(X_5,bins=[i for i in range(1,10)])

test = np.array(dataFrame)
df1 = dataFrame
df1 = df1.loc[df1['class']==2]
df1 = np.array(df1[['clump_thickness','class']])


n,bins,patches = plt.hist(df1[:,0:1],bins=[i for i in range(1,10)])

plt.ylabel('Frequency')
plt.title("Histogram of values of Clump Thickness for benign tumors:")
plt.show()


'''plt.ylabel('Frequency')
plt.title(r'$\mathrm{Histogram\ of\ Bare Nucleoli:}$')

plt.show()'''
