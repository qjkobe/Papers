from sklearn import datasets,neighbors
from metric_learn import lmnn,itml
import My_model
import My_model_1
import numpy as np
import pandas as pd

iris = datasets.load_iris()
'''
df = pd.read_csv('height-weight.csv',header=None)
target = np.array(df[0],dtype=int)
data = np.array(df.T[1:df.shape[1]].T)
'''
perm = np.random.permutation(iris.target.size)

iris.data = iris.data[perm]
iris.target = iris.target[perm]

knn_before = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_before.fit(iris.data[:100], iris.target[:100])
print(knn_before.score(iris.data[100:], iris.target[100:]))

My_1 = My_model.python_My_ML(learn_rate=1e-2, alpha=1, max_iter=100)
My_1.fit(iris.data[:100], iris.target[:100])
sets = My_1.transform(iris.data)
knn_after = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_after.fit(sets[:100], iris.target[:100])
print(knn_after.score(sets[100:], iris.target[100:]))


My_2 = My_model_1.python_My_ML(learn_rate=1e-2, alpha=1, max_iter=100)
My_2.fit(iris.data[:100], iris.target[:100])
sets = My_2.transform(iris.data)
knn_after = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_after.fit(sets[:100], iris.target[:100])
print(knn_after.score(sets[100:], iris.target[100:]))

# label_inds = np.unique(iris.target[:100], return_inverse=True)



'''
print("Improve ",(knn_after.score(test_sets,iris.target[100:])-
       knn_before.score(iris.data[100:],iris.target[100:]))/\
      knn_before.score(iris.data[100:],iris.target[100:])*100,"%")
'''




