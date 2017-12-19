from __future__ import print_function, absolute_import
import numpy as np
import math
from metric_learn import lmnn
from collections import Counter
from numpy.linalg import inv,cholesky


class python_My_ML(object):
    def __init__(self, min_iter=50, max_iter=1000, learn_rate=1e-2, regularization=0.5,
                 alpha=1):
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.regularization = regularization
        self.alpha = alpha

    def metric(self):
        L = self.transformer()
        return L.T.dot(L)

    def get_params(self):
        return self.get_params

    def transformer(self):
        return self.L

    def _process_inputs(self, X, labels):
        num_pts = X.shape[0]
        assert len(labels) == num_pts
        unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        self.labels = np.arange(len(unique_labels))
        self.X = X

        # self.L = np.eye(X.shape[1])
        model_lmnn = lmnn.python_LMNN(k=4, min_iter=50, learn_rate=1e-7)
        model_lmnn.fit(X=X[:100], labels=labels[:100])
        self.L = model_lmnn.transformer()
        # self.L = cholesky(inv(np.cov(X.T))).T
    def metric(self):
        return self.L.T.dot(self.L)

    def transform(self, X=None):
        if X is None:
            X = self.X
        return self.L.dot(X.T).T

    def m_ij_L(self, index_i, index_j, ):
        Lx = self.L.dot((index_i - index_j).T)
        return 1 - self.alpha * math.exp(- Lx.T.dot(Lx))

    def m_i_gama_L(self, X, labels, index_x):
        """
        :param X: input points
        :param labels: game^q's label
        :param index_x:point of x's index
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
                Lx = np.dot(self.L, (X[index] - X[index_x]).T)
                dic[class_index] *= (1 - self.alpha * np.math.exp(- np.dot(Lx, Lx)))
            else:
                continue
        return dic

    def partial_derivative(self, X, labels, q, num_pts, sample):
        sum = 0
        for j in range(num_pts):
            if labels[j] == q:
                for l in range(num_pts):
                    mul = 1
                    if labels[l] == q and l != j:
                        mul *= self.m_ij_L(index_i=X[sample], index_j=X[j])
                var = 2 * (1 - self.m_ij_L(index_i=X[sample], index_j=X[j])) * self.L.dot(np.outer((X[sample]-X[j]), (X[sample]-X[j])))
                sum += var * mul
            else:
                continue
        return sum

    def inner(self, X, labels, num_pts, num_labels):
        sum = 0
        for sample in range(num_pts):
            dict = self.m_i_gama_L(X=X, labels=labels, index_x=sample)
            for q in range(num_labels):
                if labels[sample] == q:
                    sum += 2 * dict[q] * self.partial_derivative(X=X, labels=labels, q=q, num_pts=num_pts, sample=sample)
                else:
                    continue
        return sum

    def inter(self, X, labels, num_pts, num_labels):
        sum = 0
        for sample in range(num_pts):
            dict = self.m_i_gama_L(X=X, labels=labels, index_x=sample)
            for q in range(num_labels):
                if labels[sample] == q:
                    mul_before = 1
                    for r in range(num_labels):
                        if r != q:
                            mul_before *= dict[r]
                        else:
                            continue
                    sub_sum = 0
                    for r in range(num_labels):
                        if r != q:
                            mul_after = 1
                            for s in range(num_labels):
                                if s != r:
                                    mul_after *= dict[s]
                                else:
                                    continue
                            sub_sum -= self.partial_derivative(X=X, labels=labels, q=r, num_pts=num_pts, sample=sample) * mul_after
                        else:
                            continue
                else:
                    continue
            sum += 2 * (1 - mul_before) * sub_sum
        return sum

    def fit(self, X, labels):
        reg = self.regularization
        learn_rate = self.learn_rate
        min_iter = self.min_iter
        self._process_inputs(X, labels)

        num_pts = X.shape[0]
        assert len(labels) == num_pts
        unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        num_labels = unique_labels.shape[0]

        for it in range(1, self.max_iter):
            print("iter:", it)
            obj_sum = 0
            for i in range(num_pts):
                obj_value = 0
                dict = self.m_i_gama_L(X=X, labels=labels, index_x=i)
                for ii in range(num_labels):
                    # ii 代表 类别 q
                    if labels[i] == ii:
                        ojb_inner = pow(dict[ii], 2)
                        for r_1 in range(num_labels):
                            leicheng_1 = 1
                            if r_1 != ii:
                                leicheng_1 *= dict[r_1]
                        ojb_inter = pow(1 - leicheng_1, 2)
                        obj_value = (1 - reg) * ojb_inner + reg * ojb_inter
                        obj_value /= 2
                obj_sum += obj_value
            obj = obj_sum
            if obj_sum > obj:
                break
            print(obj_sum)
            dG = (1 - reg) * self.inner(X=X, labels=labels, num_pts=num_pts, num_labels=num_labels) + \
                reg * self.inter(X=X, labels=labels, num_pts=num_pts, num_labels=num_labels)
            self.L -= learn_rate * dG

        return self


