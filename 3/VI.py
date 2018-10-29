import numpy as np
import pandas as pd

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

        # initialize alpha based on distributions
        self.alpha0 = np.random.gamma(shape=self.a0, scale=1 / self.b0, size=dim)
        self.sigma0 = np.diag(self.alpha0)
        self.mu0 = 0
        self.w0 = np.random.multivariate_normal(mean=np.full(dim, self.mu0), cov=self.sigma0)
        self.alphat, self.mut, self.sigmat = self.alpha0, self.mu0, self.sigma0


