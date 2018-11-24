import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import pandas as pd


class EM:
    def __init__(self, X, K):
        self.x, self.N = X, len(X)

        theta, pi = np.random.rand(K), np.random.rand(K)
        self.theta, self.pi = np.exp(theta) / np.sum(np.exp(theta)), np.exp(pi) / np.sum(np.exp(pi))
        self.pi_tmp = np.zeros([self.N, K])

        self.likelihood = 0

    def train(self, num_iter, early_stop=False):
        likelihood = []
        for _ in range(num_iter):
            prev = [self.theta, self.pi]
            self.e_step()
            self.m_step_pi()
            self.e_step()
            self.m_step_theta()
            self.cal_log_likelihood()
            likelihood.append(self.likelihood)
            if early_stop and np.isclose(prev[0], self.theta) and np.isclose(prev[1], self.pi):
                break
        return likelihood

    def e_step(self):
        for i, xx in enumerate(self.x):
            tmp = comb(20, xx) * (self.theta ** xx) * ((1 - self.theta) ** (20 - xx)) * self.pi
            self.pi_tmp[i] = tmp / sum(tmp)

    def m_step_pi(self):
        self.pi = np.sum(self.pi_tmp, axis=0) / self.N

    def m_step_theta(self):
        self.theta = np.sum(self.pi_tmp * self.x[:, None], axis=0) / np.sum(self.pi_tmp * 20, axis=0)

    def cal_log_likelihood(self):
        self.likelihood = 0
        for xx in self.x:
            tmp = comb(20, xx) * (self.theta ** xx) * ((1 - self.theta) ** (20 - xx)) * self.pi
            self.likelihood += np.log(np.max(tmp)) / self.N

    def predict(self, xx):
        return np.argmax(comb(20, xx) * (self.theta ** xx) * ((1 - self.theta) ** (20 - xx)) * self.pi)


if __name__ == '__main__':
    data = pd.read_csv("x.csv", header=None)
    xs = data[0].values
    '''
    ems = []
    KS = [3, 9, 15]
    for K in KS:
        em = EM(xs, K)
        values = em.train(50)
        ems.append(em)
        plt.plot(np.arange(2, 50, 1), values[2:], label="K={}".format(K))
        plt.legend()
    plt.savefig("1_b")
    plt.show()
    for i, em in enumerate(ems):
        r = []
        for j in range(21):
            r.append(em.predict(j))
        plt.plot(np.arange(21), r, 'bo')
        plt.title("1_c_K={}".format(KS[i]))
        plt.savefig("1_c_K={}".format(KS[i]))
        plt.show()
    '''
