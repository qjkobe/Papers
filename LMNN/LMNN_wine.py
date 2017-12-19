from sklearn import neighbors
from metric_learn import lmnn
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time
import My_model
time_begin = time.time()

df = pd.read_csv('Data_Wine.csv',header=None)
print(df.shape)
target = np.array(df[0],dtype=int)
data = np.array(df.T[1:df.shape[1]].T)

min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(data) # 数据归一化

perm = np.random.permutation(target.size)
data =data_minmax[perm]
#ata =data[perm]
target = target[perm]

knn_before = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_before.fit(data[:110],target[:110])
#print("knn_before:",knn_before.score(data[110:],target[110:]))

'''
My_model
'''
my_model = My_model.python_My_ML(learn_rate=100,alpha=1,max_iter=5)
my_model.fit(data[:110],target[:110])
train_sets_my_model = my_model.transform(data[:110])
test_sets_my_model = my_model.transform(data[110:])

knn_my_model = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_my_model.fit(train_sets_my_model,target[:110])
print(my_model.metric())
print("knn_before:",knn_before.score(data[110:],target[110:]))
print("knn_after:",knn_my_model.score(test_sets_my_model,target[110:]))
print("my_model improve:",(knn_my_model.score(test_sets_my_model,target[110:])-
       knn_before.score(data[110:],target[110:]))/\
      knn_before.score(data[110:],target[110:])*100,"%")


'''
My_model
'''
my_model_100 = My_model.python_My_ML(learn_rate=1e-5,alpha=1,max_iter=5)
my_model_100.fit(data[:110],target[:110])
train_sets_my_model_100 = my_model_100.transform(data[:110])
test_sets_my_model_100 = my_model_100.transform(data[110:])

knn_my_mode_100 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_my_mode_100.fit(train_sets_my_model_100,target[:110])
print(my_model_100.metric())
print("knn_before:",knn_before.score(data[110:],target[110:]))
print("knn_after:",knn_my_mode_100.score(test_sets_my_model_100,target[110:]))
print("my_mode_100 improve:",(knn_my_mode_100.score(test_sets_my_model_100,target[110:])-
       knn_before.score(data[110:],target[110:]))/\
      knn_before.score(data[110:],target[110:])*100,"%")

'''
LMNN
'''
lmnn = lmnn.python_LMNN(k=3,learn_rate=1e-6)
#lmnn =  My_model.python_My_ML(learn_rate=1,alpha=1,max_iter=10)
lmnn.fit(data[:110],target[:110])
#print(lmnn.metric())
train_sets_lmnn = lmnn.transform(data[:110])
test_sets_lmnn = lmnn.transform(data[110:])

knn_lmnn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_lmnn.fit(train_sets_lmnn,target[:110])
print(lmnn.metric())
print("knn_before:",knn_before.score(data[110:],target[110:]))
print("knn_after:",knn_lmnn.score(test_sets_lmnn,target[110:]))
print("lmnn improve:",(knn_lmnn.score(test_sets_lmnn,target[110:])-
       knn_before.score(data[110:],target[110:]))/\
      knn_before.score(data[110:],target[110:])*100,"%")


time_over = time.time()

print ("time cost:",time_over - time_begin)
