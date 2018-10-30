import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = "./data/"
kind_sets = ["X_set{}.csv", "y_set{}.csv", "z_set{}.csv"]


def read_one_set(num):
    paths = [data_path + p.format(num) for p in kind_sets]
    return [pd.read_csv(path, header=None) for path in paths]


class VI:
    def __init__(self, dim):
        self.dim = dim
        self.a0, self.b0 = 1e-16, 1e-16
        self.e0, self.f0 = 1., 1.
        # initialize parameters for updates
        self.at, self.bt, self.et, self.ft = np.full(dim, self.a0), np.full(dim, self.b0), self.e0, self.f0

        # initialize Sigma in some way
        self.alpha0 = np.random.gamma(1, 1, size=dim).astype(np.float64)
        self.sigma0 = np.linalg.inv(np.diag(self.alpha0))
        #self.sigma0 = np.diag(np.ones(dim, dtype=np.float64))
        self.mu0 = np.zeros([dim, 1], dtype=np.float64)
        self.mut, self.sigmat = self.mu0, self.sigma0

        # initialize data parameters
        self.x, self.y, self.N = None, None, 0

    def train(self, in_x, in_y, epoch):
        self.x, self.y = in_x.values, in_y.values
        self.N = float(self.x.shape[0])
        elbo = []
        self.cal_const_mid_results()
        for i in range(epoch):
            self.cal_mid_results()
            self.update_lambda()
            self.update_alpha()
            self.update_w()
            elbo.append(self.cal_elbo())
        return elbo

    def predict(self, x):
        w_hat = self.mut
        return np.matmul(x, w_hat)

    def cal_const_mid_results(self):
        self.xx = self.x.T.dot(self.x)
        # one bug fixed here, delete diag of xx
        self.yx = np.matmul(self.x.T, self.y)

    def cal_mid_results(self):
        self.y_wx_2 = np.square(self.y - np.matmul(self.x, self.mut))
        self.xsx = np.matmul(self.x.dot(self.sigmat), self.x.T)

    def update_lambda(self):
        self.et = self.e0 + self.N / 2
        self.ft = self.f0 + np.sum(self.y_wx_2) / 2 + np.trace(self.xsx) / 2
        # fixed a bug here from sum to trace

    def update_w(self):
        r = (self.et / self.ft)
        self.sigmat = np.linalg.inv(np.diag(self.at / self.bt) + r * self.xx)
        # fixed a bug here from sum xx to xx (D, D)
        self.mut = r * np.matmul(self.sigmat, self.yx).reshape([self.dim, 1])
        # fixed a bug here, reshape to 2d rather than 1d

    def update_alpha(self):
        self.at = np.full(self.dim, self.a0) + 1 / 2
        self.bt = self.b0 + (np.diag(self.sigmat) + self.mut.reshape(self.dim) ** 2) / 2

    def cal_elbo(self):
        r = (self.et / self.ft)
        _, logdev = np.linalg.slogdet(self.sigmat)
        E_pw = np.trace(np.diag(np.log(self.bt)) -
                        np.matmul(np.diag(self.at / self.bt), (self.sigmat + self.mut.dot(self.mut.T)))) / 2
        # fixed a bug from sum to trace here, mut**2 to mut mut^T
        elbo = (-(self.e0 + self.N / 2) * np.log(self.ft) - (r / 2) * (np.sum(self.y_wx_2) + np.trace(self.xsx))
                - r * self.f0 - E_pw - np.sum(self.a0 * np.log(self.bt) + self.b0 * self.at / self.bt) +
                logdev / 2)
        return elbo


def train_one_set(i):
    x, y, z = read_one_set(i)
    model = VI(x.shape[1])
    result = model.train(x, y, 500)
    plt.figure(num=None, figsize=(16, 9), dpi=480)
    plt.title("L of 500 iterations of data set {}".format(i))
    plt.plot(np.arange(500), result, 'g')
    plt.xlabel("Number of Iterations")
    plt.ylabel("L")
    plt.savefig('question(a)_set{}'.format(i))
    plt.show()
    return model


def question_a(test=None):
    models = []
    if test:
        model = train_one_set(test)
        return [model]
    for i in range(1, 4):
        model = train_one_set(i)
        models.append(model)
    return models


def question_b(models):
    for i, model in enumerate(models):
        tmp = model.bt / model.at
        plt.figure(num=None, figsize=(16, 9), dpi=480)
        plt.title(r"$1 / E_q[\alpha_k]$ of data set {}".format(i + 1))
        plt.plot(np.arange(model.dim), tmp, 'b')
        plt.xlabel("k")
        plt.ylabel(r"$1 / E_q[\alpha_k]$")
        plt.savefig('question(b)_set{}'.format(i + 1))
        plt.show()


def question_c(models):
    for i, model in enumerate(models):
        tmp = model.ft / model.et
        print(r"The 1 / $E_q[\lambda]$ of set{} is {}".format(i + 1, tmp))


def question_d(models, index=None):
    for i, model in enumerate(models):
        if index:
            idx = index[i]
        else:
            idx = i + 1
        x, y, z = read_one_set(idx)
        cal_y = 10 * np.sinc(z)
        pred_y = model.predict(x)
        plt.figure(num=None, figsize=(16, 9), dpi=480)
        plt.title("y values of data set {}".format(idx))
        plt.plot(z, pred_y, 'b', label='y_hat')
        plt.plot(z, y, 'ro', label='y_real')
        plt.plot(z, cal_y, 'g', label='y_calculated')
        plt.xlabel("z")
        plt.ylabel("y")
        plt.legend()
        plt.savefig('question(d)_set{}'.format(idx))
        plt.show()


if __name__ == '__main__':
    models = question_a()
    question_b(models)
    question_c(models)
    question_d(models)
