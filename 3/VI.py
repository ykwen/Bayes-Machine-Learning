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
        self.e0, self.f0 = 1, 1
        # initialize parameters for updates
        self.at, self.bt, self.et, self.ft = np.full(dim, self.a0), np.full(dim, self.b0), self.e0, self.f0

        # initialize Sigma in some way
        self.alpha0 = np.random.gamma(1, 1, size=dim)
        self.sigma0 = np.linalg.inv(np.diag(self.alpha0))
        self.mu0 = np.full(dim, 0)
        self.w0 = np.random.multivariate_normal(mean=self.mu0, cov=self.sigma0)
        self.mut, self.sigmat = self.mu0, self.sigma0

        # initialize data parameters
        self.x, self.y, self.N = None, None, 0

    def train(self, x, y, epoch):
        self.x, self.y = x.values, y.values
        self.N = float(self.x.shape[0])
        elbo = []
        self.cal_const_mid_results()
        for i in range(epoch):
            self.cal_mid_results()
            self.update_lambda()
            self.update_w()
            self.update_alpha()
            elbo.append(self.cal_elbo())
        return elbo

    def predict(self, x):
        w_hat = self.mut
        return np.matmul(x.transpose(), w_hat)

    def cal_const_mid_results(self):
        self.xx = np.diagonal(self.x.dot(self.x.T))
        self.yx = np.matmul(self.x.T, self.y)

    def cal_mid_results(self):
        self.y_wx_2 = np.square(self.y - np.matmul(self.x, self.mut))
        self.xsx = np.matmul(self.x.dot(self.sigmat), self.x.T)

    def update_lambda(self):
        self.et = self.e0 + self.N / 2
        self.ft = self.f0 + np.sum(self.y_wx_2 + self.xsx)

    def update_w(self):
        r = (self.et / self.ft)
        self.sigmat = np.linalg.inv(np.diag(self.at / self.bt) + r * np.sum(self.xx))
        self.mut = r * np.matmul(self.sigmat, self.yx).reshape(self.dim)

    def update_alpha(self):
        self.at = np.full(self.dim, self.a0) + 1 / 2
        self.bt = self.b0 + np.diagonal(self.sigmat) + self.mut ** 2

    def cal_elbo(self):
        r = (self.et / self.ft)
        elbo = (-(self.e0 + self.N / 2) * np.log(self.ft) - (r / 2) * np.sum(self.y_wx_2 + self.xsx)-r * self.f0 -
                np.sum(np.diag(np.log(self.bt)) - np.matmul(np.diag(self.at / self.bt),
                                                            (self.sigmat + self.mut ** 2))) / 2 -
                np.sum(self.a0 * np.log(self.bt) + self.b0 * self.at / self.bt) +
                np.log(np.linalg.det(self.sigmat)) / 2)
        return elbo


if __name__ == '__main__':
    x, y, z = read_one_set(1)
    model = VI(x.shape[1])
    result = model.train(x, y, 500)
    fig = plt.figure(num=None, figsize=(16, 9), dpi=480)
    plt.title("L of 500 iterations")
    plt.plot(np.arange(500), result, 'g')
    plt.xlabel("Number of Iterations")
    plt.ylabel("L")
    #plt.savefig('(a)')
    #plt.show()
    print(result)


