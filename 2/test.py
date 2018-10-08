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
