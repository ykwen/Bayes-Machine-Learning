import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, multinomial, dirichlet
from scipy.special import comb, digamma, gamma, factorial
from heapq import nlargest
import pandas as pd


class EM:
    def __init__(self, X, K):
        self.x, self.N = X, len(X)

        theta, pi = np.random.rand(K), np.random.rand(K)
        self.theta, self.pi = np.exp(theta) / np.sum(np.exp(theta)), np.exp(pi) / np.sum(np.exp(pi))
        self.phi = np.zeros([self.N, K])

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
            self.phi[i] = tmp / sum(tmp)

    def m_step_pi(self):
        self.pi = np.sum(self.phi, axis=0) / self.N

    def m_step_theta(self):
        self.theta = np.sum(self.phi * self.x[:, None], axis=0) / np.sum(self.phi * 20, axis=0)

    def cal_log_likelihood(self):
        self.likelihood = 0
        for xx in self.x:
            tmp = comb(20, xx) * (self.theta ** xx) * ((1 - self.theta) ** (20 - xx)) * self.pi
            self.likelihood += np.log(np.max(tmp)) / self.N

    def predict(self, xx):
        return np.argmax(comb(20, xx) * (self.theta ** xx) * ((1 - self.theta) ** (20 - xx)) * self.pi)


class VI:
    def __init__(self, X, K, alpha0, a0, b0):
        self.X, self.N, self.K = X, len(X), K
        self.alpha0, self.a0, self.b0 = alpha0, np.float64(a0), np.float64(b0)
        self.alpha_t, self.a_t, self.b_t = np.random.rand(K) + self.alpha0, np.repeat(self.a0, K), np.repeat(self.b0, K)
        phi = np.random.rand(self.N * K).reshape([self.N, K])
        self.phi = np.exp(phi) / np.sum(np.exp(phi), axis=1)[:, None]

    def train(self, num_iter):
        r = []
        for _ in range(num_iter):

            phi_a = np.repeat([digamma(self.a_t) - digamma(self.a_t + self.b_t)], self.N, axis=0)
            phi_b = np.repeat([digamma(self.b_t) - digamma(self.a_t + self.b_t)], self.N, axis=0)
            phi_alpha = np.repeat([digamma(self.alpha_t) - digamma(np.sum(self.alpha_t))], self.N, axis=0)
            tmp_phi = np.exp(phi_a * self.X[:, None]
                             + phi_b * (20 - self.X)[:, None] + phi_alpha)
            self.phi = tmp_phi / np.sum(tmp_phi, axis=1)[:, None]

            self.alpha_t = self.alpha0 + np.sum(self.phi, axis=0)

            self.a_t = self.a0 + np.sum(self.phi * self.X[:, None], axis=0)
            self.b_t = self.b0 + np.sum(self.phi * (20 - self.X)[:, None], axis=0)

            r.append(self.cal_l())
        return r

    def cal_l(self):
        c = np.argmax(self.phi, axis=1)
        a = np.array([self.a_t[c_i] for c_i in c])
        b = np.array([self.b_t[c_i] for c_i in c])
        alpha = np.array([self.alpha_t[c_i] for c_i in c])

        # avoid overflow or divide 0 in digamma functions
        stand = np.float64(0.0000001)
        a[a < stand], b[b < stand], c[c < stand] = stand, stand, stand
        a_t, b_t, alpha_t = self.a_t.copy(), self.b_t.copy(), self.alpha_t.copy()
        a_t[a_t < stand], b_t[b_t < stand], alpha_t[alpha_t < stand] = stand, stand, stand

        phi_a = digamma(a_t) - digamma(a_t + b_t)
        phi_b = digamma(b_t) - digamma(a_t + b_t)
        phi_a_x = digamma(a) - digamma(a + b)
        phi_b_x = digamma(b) - digamma(a + b)
        phi_alpha, phi_alpha0 = digamma(alpha), digamma(np.sum(alpha_t))
        p_x = np.sum(phi_a_x * self.X + phi_b_x * (20 - self.X))
        p_c = np.sum(digamma(alpha)) - digamma(np.sum(self.alpha_t))
        p_theta = np.sum((self.a0 - 1) * phi_a + (self.b0 - 1) * phi_b)
        p_pi = np.sum((self.alpha0 - 1) * (phi_alpha - phi_alpha0))

        q_pi = np.sum(dirichlet.entropy(self.alpha_t))
        q_theta = np.sum(beta.entropy(self.a_t, self.b_t))
        q_c = np.sum(
            -20 * np.log(factorial(20)) - \
            20 * np.sum(self.phi * np.log(self.phi), axis=1) + \
            self.multi()
        )

        '''
        q_pi = np.sum(
            np.log(np.prod(gamma(alpha_t)) / gamma(np.sum(alpha_t))) + \
            (np.sum(alpha_t) - self.K) * digamma(np.sum(alpha_t)) - \
            np.sum((alpha_t - 1) * digamma(alpha_t))
        )
        q_theta = np.sum(
            np.log(gamma(a_t) * gamma(b_t) / gamma(a_t + b_t)) - \
            (a_t - 1) * digamma(a_t) - (b_t - 1) * digamma(b_t) + \
            (a_t + b_t - 2) * digamma(a_t + b_t)
        )
        q_c = np.sum(multinomial.entropy(20, self.phi))
        '''
        if any(np.isnan(np.array([p_x, p_c, p_theta, p_pi, q_pi, q_theta, q_c]))):
            print(self.K)
            print(p_x, p_c, p_theta, p_pi, q_pi, q_theta, q_c)
        return p_x + p_c + p_theta + p_pi - q_pi - q_theta - q_c

    def multi(self):
        r = np.zeros([self.N, self.K])
        comb_result = comb(np.repeat(20, 21), np.arange(21))
        log_j = np.log(factorial(np.arange(21)))
        for i in range(21):
            r += comb_result[i] * log_j[i] * (self.phi ** i) * ((1 - self.phi) ** (20 - i))
        return np.sum(r, axis=1)

    def predict(self, xs):
        n = len(xs)
        phi_a = np.repeat([digamma(self.a_t) - digamma(self.a_t + self.b_t)], n, axis=0)
        phi_b = np.repeat([digamma(self.b_t) - digamma(self.a_t + self.b_t)], n, axis=0)
        phi_alpha = np.repeat([digamma(self.alpha_t) - digamma(np.sum(self.alpha_t))], n, axis=0)
        tmp_phi = np.exp(phi_a * xs[:, None]
                         + phi_b * (20 - xs)[:, None] + phi_alpha)
        return np.argmax(tmp_phi, axis=1)


class Gibbs:
    def __init__(self, X, K, alpha0, a0, b0):
        self.X, self.N, self.K = X, len(X), K
        self.alpha0, self.a0, self.b0 = alpha0, np.float64(a0), np.float64(b0)
        self.alpha_t, self.a_t, self.b_t = np.repeat(self.alpha0, K), np.repeat(self.a0, K), np.repeat(self.b0, K)

        self.c = np.zeros(self.N).astype(np.float64)
        theta, phi = np.random.rand(K), np.random.rand(self.N * K).reshape([self.N, K])
        self.theta, self.phi = np.exp(theta) / np.sum(np.exp(theta)), np.exp(phi) / np.sum(np.exp(phi), axis=1)[:, None]

    def train(self, num_iters):
        clusters, num = np.zeros([6, num_iters]), np.zeros(num_iters)
        # generate phi
        for itr in range(num_iters):
            unique, counts = np.unique(self.c, return_counts=True)
            # re-index
            unique = [j for j in range(len(unique))]
            x_theta = self.calculate_p_x()
            num_unique = len(unique)
            self.phi[:, :num_unique] = x_theta[:, :num_unique] * counts / (self.alpha0 + self.N - 1)
            self.phi[:, num_unique] = (self.alpha0 / (self.alpha0 + self.N - 1)) * gamma(
                self.X + self.a0) * gamma(np.float64(20) - self.X + self.b0) / gamma(
                self.a0 + np.float64(20) + self.b0
            )

            # confirm that all the others to 0
            self.phi[:, num_unique + 1:] = np.zeros_like(self.phi[:, num_unique + 1:])
            self.phi = self.phi / np.sum(self.phi, axis=1)[:, None]
            self.c = np.argmax(np.array([np.random.multinomial(1, phi_i) for phi_i in self.phi]), axis=1)

            # record clusters with n
            unique, counts = np.unique(self.c, return_counts=True)
            num_unique = len(unique)
            if num_unique >= 6:
                clusters[:, itr], num[itr] = nlargest(6, counts), num_unique
            else:
                clusters[:num_unique, itr], num[itr] = nlargest(num_unique, counts), num_unique

            # generate theta
            self.a_t, self.b_t = np.repeat(self.a0, self.K), np.repeat(self.b0, self.K)
            for ii, c_i in enumerate(self.c):
                self.a_t[c_i], self.b_t[c_i] = self.a0 + self.X[ii], self.b0 + 20 - self.X[ii]
            self.theta = np.random.beta(self.a_t, self.b_t)

        return clusters, num

    def calculate_p_x(self):
        x_theta = np.zeros([self.N, self.K]).astype(np.float64)
        for i, xx in enumerate(self.X):
            x_theta[i] = comb(20, xx) * (self.theta ** xx) * ((1 - self.theta) ** (20 - xx))
        return x_theta


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
        plt.plot(np.arange(1, 50, 1), values[1:], label="K={}".format(K))
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
    
    ks = [3, 15, 50]
    vis = []
    for K in ks:
        vi = VI(xs, K, alpha0=0.1, a0=0.5, b0=0.5)
        values = vi.train(1000)
        vis.append(vi)
        plt.plot(np.arange(1, 1000, 1), values[1:], label="K={}".format(K))
        plt.legend()
    plt.savefig("2_b")
    plt.show()
    for i, vi in enumerate(vis):
        values = vi.predict(np.arange(21))
        plt.plot(np.arange(21), values, "bo", label="K={}".format(ks[i]))
        plt.legend()
        plt.savefig("2_c_K={}".format(ks[i]))
        plt.show()
    '''
    gibbs = Gibbs(xs, 30, 0.75, 0.5, 0.5)
    t = 1000
    clusters, num = gibbs.train(t)
    for i, cluster in enumerate(clusters):
        plt.plot(np.arange(t), cluster, label="cluster_{}".format(i+1))
    plt.legend()
    plt.savefig("3_b")
    plt.show()
    plt.title("number of clusters")
    plt.plot(np.arange(t), num)
    plt.savefig("3_c")
    plt.show()



