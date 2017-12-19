from sklearn import neighbors
from metric_learn import lmnn
import  My_model
import numpy as np
import pandas as pd
import time

time_begin = time.time()

df = pd.read_csv('abalone_data.csv',header=None)

target = np.array(df[df.shape[1]-1],dtype=int)
data = np.array(df.T[:df.shape[1]-1].T)

perm = np.random.permutation(target.size)

data =data[perm]
target = target[perm]

lmnn = My_model.python_My_ML(learn_rate=1e-7,alpha=0.5)


lmnn.fit(data[:3333],target[:3333])

train_sets = lmnn.transform(data[:3333])
test_sets = lmnn.transform(data[3333:])

knn_before = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_before.fit(data[:3333],target[:3333])

print(knn_before.score(data[3333:],target[3333:]))

knn_after = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_after.fit(train_sets,target[:3333])

print(knn_after.score(test_sets,target[3333:]))

print (knn_after.score(test_sets,target[3333:])-
       knn_before.score(data[3333:],target[3333:]))/\
      knn_before.score(data[3333:],target[3333:])*100

time_over = time.time()

print(time_over - time_begin)

