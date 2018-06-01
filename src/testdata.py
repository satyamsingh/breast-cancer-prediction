from sklearn.datasets import load_iris
from sklearn import tree

from subprocess import check_call


clf = tree.DecisionTreeClassifier()
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf,
    out_file='tree.dot')

check_call(['dot','-Tpng','tree.dot','-o','OutputFile.png'])
