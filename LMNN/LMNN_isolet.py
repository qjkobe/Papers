from sklearn import neighbors
from metric_learn import lmnn
from sklearn.decomposition import PCA
import My_model
import numpy as np
import pandas as pd
import time

time_begin = time.time()

train = pd.read_csv('DATA_isolet_train.csv',header=None)
test = pd.read_csv('DATA_isolet_test.csv',header=None)

train_target = np.array(train[train.shape[1]-1],dtype=int)
test_target = np.array(test[test.shape[1]-1],dtype=int)

train_data = np.array(train.T[:train.shape[1]-1].T)
test_data = np.array(test.T[:test.shape[1]-1].T)

pca = PCA(n_components=2)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)
lmnn = My_model.python_My_ML(learn_rate=1e-7,alpha=0.5,max_iter=100)
#lmnn = lmnn.python_LMNN(k=3,learn_rate=1e-6)
lmnn.fit(train_data,train_target)

train_set = lmnn.transform(train_data)
test_set = lmnn.transform(test_data)

knn_before = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_before.fit(train_data,train_target)
print( knn_before.score(test_data,test_target))


knn_after = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_after.fit(train_set,train_target)
print(knn_after.score(test_set,test_target))

print (knn_after.score(test_set,test_target) - knn_before.score(test_data,test_target))/ knn_before.score(test_data,test_target)

time_over = time.time()

print(time_over - time_begin)
