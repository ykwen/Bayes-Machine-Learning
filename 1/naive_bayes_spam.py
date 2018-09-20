import pandas as pd
import numpy as np
import scipy.special as sp
import scipy.stats as st
import math

label_train = pd.read_csv('label_train.csv', header=None)
label_test = pd.read_csv('label_test.csv', header=None)
x_train = pd.read_csv('x_train.csv', header=None)
x_test = pd.read_csv('x_test.csv', header=None)

feature_number = x_train.shape[1]
num_train = x_train.shape[0]
num_test = x_test.shape[0]


class NBClassifier:
    def __init__(self, dim, num_class, a=1, b=1, e=1, f=1):
        self.dim = dim
        self.c = num_class
        self.r = np.zeros([self.c, self.dim])
        self.a, self.b, self.e, self.f = a, b, e, f
        self.py = np.zeros(2)

    def train(self, xs, ys):
        self.py[0] = 1 /(self.b + ys[ys[0] == 0].shape[0] + 1)
        self.py[1] = 1 /(self.b + ys[ys[0] == 1].shape[0] + 1)

        x_zeros = xs[ys[0] == 0]
        x_ones = xs[ys[0] == 1]
        self.r[0], self.r[1] = self.cal_r(x_zeros), self.cal_r(x_ones)

    def cal_r(self, xs):
        result = np.zeros(self.dim)
        for i in range(self.dim):
            result[i] = xs[i].sum()
        return result

    def predict(self, px):
        prb = np.full(self.c, 1.0).astype(np.float64)
        for i in range(self.c):
            for j in range(self.dim):
                prb[i] += np.log(st.nbinom.pmf(k=px[j], n=self.r[i][j], p=1-self.py[i]))
        return prb


nbc = NBClassifier(feature_number, 2)
nbc.train(x_train, label_train)
prob = np.zeros([num_test, 2], dtype=np.float64)
bad_case = []
for i, x in enumerate(x_test.values):
    p = nbc.predict(x)
    prob[i] = p

prob = [[a/(a+b), b/(a+b)]for a, b in prob]
pred = np.array([a > b for a, b in prob]).astype(int)


def cal_result(pred_y, label):
    result = np.zeros([2, 2])
    for p, l in zip(pred_y, label.values):
        result[l[0]][p] += 1
    return result


print(cal_result(pred, label_test))
### Need To Improve the Comparation Process