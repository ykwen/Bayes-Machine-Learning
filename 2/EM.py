import numpy as np
import pandas as pd
import numpy.ma as mask
from scipy.stats import norm
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
U = np.random.normal(0, ini_scale, (num_users, d))
V = np.random.normal(0, ini_scale, (num_movies, d))

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
        # Update parameters
        self.M_step(expectation)
        # Check output
        return self.cal_lnP()

    def E_step(self):
        # Calculate u_i dot v_j in matrix form
        prod = np.matmul(self.U, self.V.T)
        # Calculate CDF and PDF of product/sigma^2
        cdf = norm.cdf(-prod)
        pdf = norm.pdf(-prod)
        # Calculate r_ij is 1 and -1 separately using masks, the true value will be masked
        mask_pos = self.R == -1
        mask_neg = self.R == 1
        # Calculate expectations
        pos = self.sig * pdf / (1-cdf)
        neg = - self.sig * pdf / cdf
        return prod + mask.masked_array(pos, mask_pos) + mask.masked_array(neg, mask_neg)

    def M_step(self, E):
        return

    def cal_lnP(self):
        return


model = EM(R_train, U, V, d, c, sigma_square)
print(model.E_step())