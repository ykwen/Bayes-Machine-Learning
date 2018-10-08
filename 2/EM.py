import numpy as np
import pandas as pd
import numpy.ma as mask
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('ratings.csv', header=None, names=['user_id', 'movie_id', 'rate'])
test_data = pd.read_csv('ratings_test.csv', header=None, names=['user_id', 'movie_id', 'rate'])

# Load movie names with index
with open('movies.txt', 'rb') as f:
    movies = f.readlines()

# Extract data features
num_users = train_data['user_id'].nunique()
num_movies = len(movies)
## Validation check
if test_data['user_id'].nunique() > num_users or test_data['movie_id'].nunique() > num_movies:
    print("Warning, the test data has user of movie not presented in train data")
    quit()

# Set parameters as required
d = 5
c = 1
sigma_square = 1
ini_sig_sq = 0.1 # the parameter used to initialize

# Initialize U and V
ini_scale = np.sqrt(ini_sig_sq)
U = np.random.normal(0, ini_scale, (num_users, d)).astype(np.float64)
V = np.random.normal(0, ini_scale, (num_movies, d)).astype(np.float64)

# Build rating matrix for each user and movie, leave nan for not presented
R_train = np.full([num_users, num_movies], np.nan)
for index, row in train_data.iterrows():
    R_train[row[0]-1][row[1]-1] = row[2] # ID - 1 to fit the index of matrix


# Implement EM algorithm
class EM:
    def __init__(self, R, U, V, d, c, sig_square, param_distr='normal'):
        self.R = R
        self.U = U
        self.V = V
        self.d = d
        self.c = c
        self.sig_square = sig_square
        self.sig = np.sqrt(self.sig_square)
        if param_distr != 'normal':
            print("This type of parameter is not supported now.")
            quit()

    # One single iteration
    def iter(self):
        # Calculate expectation
        expectation = self.E_step()
        # Update U
        update_U = True
        self.M_step(expectation, update_U)
        # Calculate expectation
        expectation = self.E_step()
        # Update V
        update_U = False
        self.M_step(expectation, update_U)
        # Check output
        return self.cal_lnP()

    def E_step(self):
        # Calculate u_i dot v_j in matrix form
        prod = np.matmul(self.U, self.V.T) / self.sig
        # Calculate CDF and PDF of product/sigma^2
        cdf = norm.cdf(-prod)
        pdf = norm.pdf(-prod)
        # Calculate r_ij is 1 and -1 separately using masks, the true value will be masked
        mask_pos = self.R == -1
        mask_neg = self.R == 1
        # Calculate expectations
        pos = self.sig * pdf / (1-cdf)
        neg = - self.sig * pdf / cdf
        return prod + mask.masked_array(pos, mask_pos).filled(0) + mask.masked_array(neg, mask_neg).filled(0)

    # Use flag to determine whether U or V is updated
    def M_step(self, E, flag=True):
        if flag:
            # Calculate optimized U
            inversed = np.linalg.inv(np.identity(d) / self.c + np.matmul(self.V.T, self.V) / self.sig_square)
            prod_e = np.matmul(E, self.V) / self.sig_square
            new_U = np.matmul(prod_e, inversed)
            # Update U
            self.U = new_U
        else:
            # Calculate optimized V
            inversed = np.linalg.inv(np.identity(d) / self.c + np.matmul(self.U.T, self.U) / self.sig_square)
            prod_e = np.matmul(E.T, self.U) / self.sig_square
            new_V = np.matmul(prod_e, inversed)
            # Update V
            self.V = new_V
        return

    def cal_lnP(self):
        # Calculate each part to calculate lnp(R,U,V)
        const = -(self.U.shape[0] + self.V.shape[0]) * np.log(2 * np.pi * self.c)
        mat_norm_square = - (np.trace(np.inner(self.U, self.U)) + np.trace(np.inner(self.V, self.V))) / (2 * self.c)
        ### Calculate bernoulli distribution
        cdf = norm.cdf(np.matmul(self.U, self.V.T) / self.sig)
        mask_pos, mask_neg = self.R == -1, self.R == 1
        pos, neg = np.log(cdf), np.log(1-cdf)
        return const + mat_norm_square + np.sum(mask.masked_array(pos, mask_pos).filled(0)) +\
               np.sum(mask.masked_array(neg, mask_neg).filled(0))

    def predict(self, user_id, movie_id):
        prod = np.matmul(self.U[user_id], self.V[movie_id].T)
        pr = norm.cdf(prod)
        if pr > 0.5:
            return 1
        else:
            return -1


'''
(a) run 100 iterations and plot iterations 2 through 100
'''
model = EM(R_train, U, V, d, c, sigma_square)
result = []
for i in range(100):
    result.append(model.iter())
fig = plt.figure(num=None, figsize=(16, 9), dpi=480)
plt.title("lnp(R,U,V) from iteration 2 through 100")
plt.plot(np.arange(99), result[1:], 'go')
plt.xlabel("Number of Iterations")
plt.ylabel("lnp(R,U,V)")
plt.savefig('(a)')
plt.show()

'''
(b) show 5 starting point
'''
result = []
for i in range(5):
    r = []
    U = np.random.normal(0, ini_scale, (num_users, d)).astype(np.float64)
    V = np.random.normal(0, ini_scale, (num_movies, d)).astype(np.float64)
    model = EM(R_train, U, V, d, c, sigma_square)
    for _ in range(100):
        r.append(model.iter())
    result.append(r)
    print("Finish the %s-th model" % (i+1))
fig = plt.figure(num=None, figsize=(16, 9), dpi=480)
plt.title("lnp(R,U,V) from iteration 2 through 100")
colors = {0: 'b', 1: 'g', 2: 'c', 3: 'm', 4: 'y'}
for i in range(5):
    plt.plot(np.arange(80), result[i][20:], colors[i]+'o', label=str(i))
plt.xlabel("Number of Iterations")
plt.ylabel("lnp(R,U,V)")
plt.legend()
plt.savefig('(b)')
plt.show()

'''
(c) Predict values with confusion matrix
'''
max_acc = 0
for _ in range(5):
    # Initialize and train model
    U = np.random.normal(0, ini_scale, (num_users, d)).astype(np.float64)
    V = np.random.normal(0, ini_scale, (num_movies, d)).astype(np.float64)
    model = EM(R_train, U, V, d, c, sigma_square)
    result = []
    for i in range(500):
        result.append(model.iter())
        if i % 100 == 0:
            print("At iteration %s" % i)
    # Predict and count result
    count = np.zeros([2, 2])
    for i, row in test_data.iterrows():
        p = model.predict(row[0]-1, row[1]-1)  # minus one from id to index
        label, p = max(0, row[2]), max(0, p)
        count[label][p] += 1
    print(count)
    acc = (count[0][0] + count[1][1]) / np.sum(count)
    if acc > max_acc:
        pickle.dump(model, open('best_model.pkl', 'wb+'))
        result = pd.DataFrame({'predicted_-1': count[:, 0], 'predicted_1': count[:, 1]})
        result.to_csv('best_result.csv')
