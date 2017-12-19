"""
Belif funcation Metric Learning,

BFML without regularization terms.
"""
from __future__ import print_function, absolute_import
import numpy as np
import math
from metric_learn import lmnn
from metric_learn import RCA_Supervised
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from numpy.linalg import inv,cholesky

class BaseMetricLearner(object):
    def __init__(self):
        raise NotImplementedError('BaseMetricLearner should not be instantiated')

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix.

		.. math:: M = L^{\\top} L

		Returns
		-------
		M : (d x d) matrix
		"""
        L = self.transformer()
        return L.T.dot(L)

    def transformer(self):
        """Computes the transformation matrix from the Mahalanobis matrix.

        L = inv(cholesky(M))

        Returns
        -------
        inv():qiu ni
        cholesky():fenjie matrix
        L : (d x d) matrix
        """
        return inv(cholesky(self.metric()))

    def transform(self, X=None):
        """Applies the metric transformation.

		Parameters
		----------
		X : (n x d) matrix, optional
			Data to transform. If not supplied, the training data will be used.

		Returns
		-------
		transformed : (n x d) matrix
			Input data transformed to the metric space by :math:`XL^{\\top}`
		"""
        if X is None:
            X = self.X
        L = self.transformer()
        return X.dot(L.T)
        # return self.L.dot(X.T).T

    def get_params(self, deep=False):
        """Get parameters for this metric learner.

		Parameters
		----------
		deep: boolean, optional
			@WARNING doesn't do anything, only exists because
			scikit-learn has this on BaseEstimator.

		Returns
		-------
		params : mapping of string to any
			Parameter names mapped to their values.
        :param self:
		"""
        return self.params

    def set_params(self, **kwarg):
        """Set the parameters of this metric learner.

		Overwrites any default parameters or parameters specified in constructor.

		Returns
		-------
		self
		"""
        self.params.update(kwarg)
        return self


# commonality between My_ML implementations
class _base_My_ML(BaseMetricLearner):
    def __init__(self, **kwargs):
        self.params = kwargs

    def transformer(self):
        return self.L

class python_My_ML(_base_My_ML):
    def __init__(self, min_iter=50, max_iter=1000, learn_rate=1e-2, regularization=0.5,
                 alpha=1, convergence_tol=0.001, verbose=False):
        """Initialize the ML object
        k: number of neighbors to consider. (does not include self-edges)
        regularization: weighting of pull and push terms
        """
        _base_My_ML.__init__(self, min_iter=min_iter, max_iter=max_iter,
                             learn_rate=learn_rate, regularization=regularization, alpha=alpha,
                             convergence_tol=convergence_tol, verbose=verbose)

    # input data sets & init L
    def _process_inputs(self, X, labels):
        num_pts = X.shape[0]
        assert len(labels) == num_pts
        unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        self.labels = np.arange(len(unique_labels))
        self.X = X
        # min_max_scaler = preprocessing.MinMaxScaler()
        # self.X = min_max_scaler.fit_transform(X) # 数据归一化
        # model_rca =RCA_Supervised(num_chunks=100)
        # model_rca.fit(X=X, labels=labels)
        # self.L = model_rca.transformer()
        model_lmnn = lmnn.python_LMNN(k=3, min_iter=50, learn_rate=1e-7)
        model_lmnn.fit(X=X, labels=labels)
        self.L = model_lmnn.transformer()
        # self.L = np.linalg.cholesky(np.linalg.inv(np.cov(X.T)))

    # return metric M=L^T\dot L
    def metric(self):
        return self.L.T.dot(self.L)

    def transform(self, X=None):
        if X is None:
            X = self.X
        # L = self.transformer()
        return self.L.dot(X.T).T
        # return X.dot(L.T)

    def fit(self, X, labels):

        alpha = self.params['alpha']
        reg = self.params['regularization']
        # reg：正则化参数，即\mu
        learn_rate = self.params['learn_rate']
        # learn_rate：学习率

        convergence_tol = self.params['convergence_tol']
        min_iter = self.params['min_iter']
        self._process_inputs(X, labels)

        # init L
        L = self.L
        num_pts = X.shape[0]
        unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        num_labels = unique_labels.shape[0]

        # 第一层循环：控制迭代次数
        for it in range(0, self.params['max_iter']):
            print("iter:",it)
            # computer obj value
            obj_sum = 0
            for i in range(num_pts):
                obj_value = 0
                dict = m_i_gama_Omage(X, labels, L=L, index_x=i, alpha=alpha)
                for ii in range(num_labels):
                    # ii 代表 类别 q
                    if labels[i] == ii:
                        ojb_inner = pow(dict[ii], 2)
                        for r_1 in range(num_labels):
                            leicheng_1 = 1
                            if r_1 != ii:
                                leicheng_1 *= dict[r_1]
                        ojb_inter = pow(leicheng_1, 2)
                        obj_value = (1 - reg) * ojb_inner + reg * ojb_inter
                obj_sum += obj_value
            print(obj_sum)
            # 第二层循环：遍历所有样本点，求出累计梯度
            dG = np.zeros_like(L)
            for i in range(num_pts):
                # i 代表样本点i
                dict = m_i_gama_Omage(X,labels,L=L,index_x=i,alpha=alpha)
                # 第三层循环：遍历类别标签
                for ii in range(num_labels):
                    # ii 代表 类别 q
                    if labels[i] == ii:
                        # begin 类内距离
                        sum = 0
                        for jj in range(num_pts):
                            # jj 代表 属于gama_q的样本集合
                            if labels[jj]==ii :
                                m_tj = m_tj_omega(l=L, index_x=X[i], index_y=X[jj], alpha=alpha)
                                x_ij = X[i] - X[jj]
                                dot = np.outer(x_ij, x_ij)
                                Ldot = L.dot(dot)
                                sum_1 = m_tj * Ldot
                                sum += 2 * sum_1
                        # inner:类内距离
                        inner = dict[ii] * sum
                        # end 类内距离
                        # begin 类间距离
                        for r_1 in range(num_labels):
                            leicheng_1 = 1
                            if r_1 != ii:
                                leicheng_1 *= dict[r_1]
                        # 下面计算后括号内容
                        leijia_1 = 0
                        for r_2 in range(num_labels):
                            if r_2 != ii:
                                leicheng_2 = 1
                                for s in range(num_labels):
                                    if s != r_2:
                                        leicheng_2 *= dict[s]
                                for jjj in range(num_pts):
                                    # jjj 代表 属于gama_r的样本集合
                                    if labels[jjj] == r_2:
                                        leijia_1 += 2 * m_tj_omega(index_x=X[i], l=L, index_y=X[jjj], alpha=alpha) * \
                                                    L.dot(np.outer((X[i]-X[jjj]), (X[i]-X[jjj]))) * leicheng_2
                        inter = (1 - leicheng_1) * (-leijia_1)
                        # end 类间距离
                        dfG = (1 - reg) * inner + reg * inter
                    else:
                        dfG = 0
                dG += dfG
            L -= learn_rate * dG

            self.L = L
        return self


def m_i_gama_Omage(X, labels, index_x, L, alpha):
    """
    :param X: input points
    :param labels: game^q's label
    :param index_x:point of x's index
    :param L:
    :param alpha:
    :return: m_i^gama_q(Omega)
    """
    num_pts = X.shape[0]
    dic = {}
    unique_labels, label_inds = np.unique(labels, return_inverse=True)
    num_labels = unique_labels.shape[0]
    for class_index in range(num_labels):
        # class_index代表类别 q
        dic[class_index] = 1
    for index in range(num_pts):
        if index != index_x:
            class_index = labels[index]
            Lx = np.dot(L, (X[index] - X[index_x]).T)
            dic[class_index] *= (1 - alpha * np.math.exp(- np.dot(Lx, Lx)))
        else:
            continue
    # for class_index in range(num_labels):
    #     dic[class_index] /= max(dic.values())
    return dic



def m_tj_omega(l, index_x, index_y, alpha):
    lx = l.dot((index_x - index_y).T)
    x = math.exp(- lx.T.dot(lx))
    return alpha * x