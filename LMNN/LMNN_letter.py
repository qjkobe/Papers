import time
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),":The program is beginning")
import numpy as np
import pandas as pd
import My_model
from sklearn import neighbors
from metric_learn import lmnn
from sklearn.decomposition import PCA



time_begin = time.time()

# data = pd.read_csv('DATA_letter_1.csv',header=None)
data = pd.read_csv('height-weight.csv',header=None)
target = np.array(data[0])
data = np.array(data.T[1:2].T,dtype=float)

perm = np.random.permutation(target.size)

data = data[perm]
target = target[perm]

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),":Lmnn is beginning ")

lmnn =My_model.python_My_ML(learn_rate=1e-5,alpha=0.5,max_iter=100)
#lmnn = lmnn.python_LMNN(learn_rate=1e-6)
lmnn.fit(data[:10],target[:10])

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),":The fit is over")

train_sets = lmnn.transform(data[:10])
test_sets = lmnn.transform(data[10:])

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),":The knn is beginning")
knn_before = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_before.fit(data[:10],target[:10])
print("Before_score:",knn_before.score(data[10:],target[10:]))


knn_after = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_after.fit(train_sets,target[:10])
print ("After_score:",knn_after.score(test_sets,target[10:]))

print ("Improve ",(knn_after.score(test_sets,target[10:]) - knn_before.score(data[10:],target[10:]))/ knn_before.score(data[10:],target[0:]) * 100,"%")

time_over = time.time()

print( "All time:",time_over - time_begin,"s!")

print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),":GG")
