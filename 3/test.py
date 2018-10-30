import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln
import matplotlib.pyplot as plt

data_path = "./data/"
kind_sets = ["X_set{}.csv", "y_set{}.csv", "z_set{}.csv"]


def read_one_set(num):
    paths = [data_path + p.format(num) for p in kind_sets]
    return [pd.read_csv(path, header=None) for path in paths]


a_0 = b_0 = 1e-16
e_0 = f_0 = 1

def update_q_alpha_k(a_0, b_0, mu, sigma, k):
    a_kp = 0.5 + a_0
    E_w_k_2 = mu[k]**2 + sigma[k, k]
    b_kp = 0.5 * E_w_k_2 + b_0
    return a_kp, b_kp

# Compute L_1
def L_1(dim, a_t, b_t, mu_t, sigma_t):
    A_t = a_t / b_t
    E_A = np.diag(A_t)
    E_w_T_w = sigma_t + np.dot(mu_t, mu_t.T)
    E_ln_alpha = map(lambda a_b: digamma(a_b[0]) - np.log(a_b[1]), zip(a_t, b_t))
    print("L_1 is p_w is " + str(0.5 * sum(E_ln_alpha) - 0.5 * np.trace(np.dot(E_w_T_w, E_A))))
    return - 0.5 * dim * np.log(2 * np.pi) + 0.5 * sum(E_ln_alpha) - 0.5 * np.trace(np.dot(E_w_T_w, E_A))

# Compute L_2
def L_2(e_0, f_0, e_t, f_t):
    print("L_2 is p_lambda is "+ str((e_0 -1) * (digamma(e_t) - np.log(f_t)) - f_0 * e_t/float(f_t)))
    return e_0 * np.log(f_0) - gammaln(e_0) + (e_0 -1) * (digamma(e_t) - np.log(f_t)) - f_0\
           * e_t/float(f_t)

# Compute L_3
def L_3(a_0, b_0, a_t, b_t):
    print("L_3 is p_alpha is "+ str(sum(map(lambda a_b: (a_0 - 1) * (digamma(a_b[0]) - np.log(a_b[1])) \
                   - b_0 * a_b[0]/float(a_b[1]), zip(a_t, b_t)))))
    return sum(map(lambda a_b: a_0 * np.log(b_0) - gammaln(a_0) + (a_0 - 1) * (digamma(a_b[0]) - np.log(a_b[1])) \
                   - b_0 * a_b[0]/float(a_b[1]), zip(a_t, b_t)))


x, y, z = read_one_set(2)
dim = x.shape[1]
n = x.shape[0]

a_t = np.array([a_0] * dim, dtype='float64')
b_t = np.array([b_0] * dim, dtype='float64')

# Compute L_4
def L_4(dim, sigma_t):
    # Compute in logspace to prevent overflow
    sign, logdet_sigma_t = np.linalg.slogdet(sigma_t)
    # As 2 * np.pi * np.exp(1))** dim) is a constant and causes overflow, remove
    return - 0.5 * (sign * logdet_sigma_t)

# Compute L_5
def L_5(e_t, f_t):
    return np.log(f_t) - gammaln(e_t) + (e_t - 1) * digamma(e_t) - e_t

# Compute L_6
def L_6(a_t, b_t):
    return sum(map(lambda a_b: np.log(a_b[1]) - gammaln(a_b[0]) + (a_b[0] - 1) * digamma(a_b[0]) - a_b[0], zip(a_t, b_t)))


# Compute L_7
def L_7(n, e_t, f_t, mu_t, sigma_t, Y_T_Y, Y_T_X_T, X_X_T):
    E_ln_lambda = digamma(e_t) - np.log(f_t)
    E_lambda = e_t / float(f_t)
    E_w_T_w = np.dot(mu_t, mu_t.T) + sigma_t
    print("L_7 is p_y is " + str(0.5 * n  * (E_ln_lambda) \
           - 0.5 * E_lambda * (Y_T_Y - 2 * np.dot(Y_T_X_T, mu_t) + np.trace(np.dot(E_w_T_w, X_X_T)))))
    return 0.5 * n * (E_ln_lambda - np.log(2 * np.pi)) \
           - 0.5 * E_lambda * (Y_T_Y - 2 * np.dot(Y_T_X_T, mu_t) + np.trace(np.dot(E_w_T_w, X_X_T)))

def update_alpha2(a_0, b_0, mu_t, sigma_t, dim):
    at = np.full(dim, a_0) + 1 / 2
    bt = b_0 + (np.diag(sigma_t) + mu_t.reshape(dim) ** 2) / 2
    return at, bt

def update_w2(e_t, f_t, a_t, b_t, dim, x, y):
    r = (e_t / f_t)
    xx = x.T.dot(x)
    yx = np.matmul(x.T, y)
    sigmat = np.linalg.inv(np.diag(a_t / b_t) + r * xx)
    mut = r * np.matmul(sigmat, yx).reshape([dim, 1])
    return mut, sigmat

# Update q(w)
def update_q_w(e_t, f_t, a_t, b_t, X_X_T, X, Y):
    A_p = a_t / b_t
    E_lambda = e_t/float(f_t)
    M_p = E_lambda * X_X_T + np.diag(A_p)
    sigma_p = np.linalg.inv(M_p)
    mu_p = E_lambda * np.dot(np.dot(sigma_p, X.T), Y)
    return mu_p, sigma_p


def update_q_lambda(n, e_0, f_0, mu, sigma, Y_T_Y, Y_T_X_T, X_X_T):
    e_p = 0.5 * n + e_0
    E_w_w_T = sigma + np.dot(mu, mu.T)
    f_p = 0.5 * (Y_T_Y - 2 * np.dot(Y_T_X_T, mu) + np.trace(np.dot(E_w_w_T, X_X_T))) + f_0
    return e_p, float(f_p)


def update_lambda2(n, e_0, f_0, mu_t, sigma_t, x, y):
    et = e_0 + n / 2
    y_wx = y - np.matmul(x, mu_t)
    y_wx_2 = y_wx ** 2
    xsx = np.matmul(sigma_t, x.T.dot(x))
    ft = f_0 + np.sum(y_wx_2)/2 + np.trace(xsx) / 2
    return et, ft

def variational_inference(X, y):
    # Get dimensions of X
    dim = X.shape[1]
    n = X.shape[0]

    # Calculate variables
    Y_T_Y = np.dot(y.T, y)[0][0]
    Y_T_X_T = np.dot(y.T, X)
    X_X_T = np.dot(X.T, X)

    # Initialise variables
    a_t = np.array([a_0] * dim, dtype='float64')
    b_t = np.array([b_0] * dim, dtype='float64')
    e_t = e_0
    f_t = f_0
    mu_t = np.zeros(dim, dtype='float64')
    sigma_t = np.diag(np.ones(dim, dtype='float64'))
    L = []
    L_1_t = L_2_t = L_3_t = L_4_t = L_5_t = L_6_t = L_7_t = 0

    # Run VI algorithm
    for t in range(500):
        # print t
        e_t, f_t = update_q_lambda(n, e_0, f_0, mu_t, sigma_t, Y_T_Y, Y_T_X_T, X_X_T)
        et2, ft2 = update_lambda2(n, e_0, f_0, mu_t, sigma_t, X, y)
        #print(f_t, ft2)
        e_t, f_t = et2, ft2

        a_p = []
        b_p = []
        for k in range(dim):
            a_kp, b_kp = update_q_alpha_k(a_0, b_0, mu_t, sigma_t, k)
            a_p.append(a_kp)
            b_p.append(b_kp)
        a_t = np.array(a_p)
        b_t = np.array(b_p).flatten()
        a_t, b_t = update_alpha2(a_0, b_0, mu_t, sigma_t, dim)

        mu_t, sigma_t = update_w2(e_t, f_t, a_t, b_t, dim, x, y)

        L_1_t = L_1(dim, a_t, b_t, mu_t, sigma_t)
        L_2_t = L_2(e_0, f_0, e_t, f_t)
        L_3_t = L_3(a_0, b_0, a_t, b_t)
        L_4_t = L_4(dim, sigma_t)
        L_5_t = L_5(e_t, f_t)
        L_6_t = L_6(a_t, b_t)
        L_7_t = L_7(n, e_t, f_t, mu_t, sigma_t, Y_T_Y, Y_T_X_T, X_X_T)
        # print L_1_t, L_2_t, L_3_t, L_4_t, L_5_t, L_6_t, L_7_t
        L.append((L_1_t + L_2_t + L_3_t + L_7_t - L_4_t - L_5_t - L_6_t)[0][0])

    return L, a_t, b_t, e_t, f_t, mu_t, sigma_t, dim

LL_1, a_1, b_1, e_1, f_1, mu_1, sigma_1, dim_1 = variational_inference(x.values, y.values)

plt.plot(range(500), LL_1)
plt.xlabel('Iterations')
plt.ylabel('Variational Objective Function')
plt.title('Variational Objective Function against Iterations (Dataset 1)')
plt.show()

plt.plot(range(dim_1), 1/(a_1/b_1))
plt.xlabel('Dimensions')
plt.ylabel(r'$1/\mathbb{E}_{q[\alpha_k]}$')
plt.title(r'$1/\mathbb{E}_{q[\alpha_k]}$ against Dimensions (Dataset 1)')
plt.show()


y_hat = np.dot(x, mu_1)
plt.plot(z, y_hat, 'r', label=r'$\hat{y}$')
plt.scatter(z, y, c='y', label='Scatter')
plt.plot(z, 10 * np.sinc(z), label='Ground Truth')
plt.legend()
plt.xlabel('z_i')
plt.ylabel('y')
plt.title('Various Ys against Z (Dataset 1)')
plt.show()
