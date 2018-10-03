import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Load files using pandas
label_train = pd.read_csv('label_train.csv', header=None)
label_test = pd.read_csv('label_test.csv', header=None)
x_train = pd.read_csv('x_train.csv', header=None)
x_test = pd.read_csv('x_test.csv', header=None)

# Read data parameters
feature_number = x_train.shape[1]
num_train = x_train.shape[0]
num_test = x_test.shape[0]


# Define the classifier
class NBClassifier:
    def __init__(self, dim, num_class, a=1, b=1, e=1, f=1):
        # Initialize with the given parameters
        self.dim = dim
        self.c = num_class
        self.r = np.zeros([self.c, self.dim])
        self.a, self.b, self.e, self.f = a, b, e, f
        self.py = np.zeros(2)
        self.pyNb = np.zeros(2)

    # Train the model by calculating distributions
    def train(self, xs, ys):
        # Calculate the distribution of y in training data
        self.py[0] = (self.f + ys[ys[0] == 0].shape[0]) / (xs.shape[0] + self.e + self.f)
        self.py[1] = (self.e + ys[ys[0] == 1].shape[0]) / (xs.shape[0] + self.e + self.f)

        # Calculate the probability for negative binomial distribution
        self.pyNb[0] = 1 / (self.b + ys[ys[0] == 0].shape[0] + 1)
        self.pyNb[1] = 1 / (self.b + ys[ys[0] == 1].shape[0] + 1)

        x_zeros = xs[ys[0] == 0]
        x_ones = xs[ys[0] == 1]
        self.r[0], self.r[1] = self.cal_r(x_zeros), self.cal_r(x_ones)

    # Calculate parameters based on distribution
    def cal_r(self, xs):
        result = np.full(self.dim, self.a, dtype=np.float64)
        for i in range(self.dim):
            result[i] += xs[i].sum()
        return result

    # Predict label by taking the sum of all features' log probability
    def predict_one(self, px):
        # Initialize log likelihood as 0
        prb = np.full(self.c, 0.0).astype(np.float64)
        for i in range(self.c):
            prb[i] += np.log(self.py[i])
            for j in range(self.dim):
                # Modified a little to fit scipy function
                prb[i] += np.log(st.nbinom.pmf(k=px[j], n=self.r[i][j], p=1-self.pyNb[i]))
        return prb

    # Calculate the parameters for plot for (c) and (d)
    def average_lambda(self, xs, ys, y):
        # Not sure here
        result = []
        xs = xs[ys[0] == y]
        for d in range(self.dim):
            a = self.a
            b = self.b
            for x in xs.values:
                a += x[d]
                b += 1
            result.append(np.divide(a, b))
        return result


nbc = NBClassifier(feature_number, 2)
nbc.train(x_train, label_train)
prob = np.zeros([num_test, 2], dtype=np.float64)
bad_case = []
for i, x in enumerate(x_test.values):
    prob[i] = nbc.predict_one(x) # Predict test data


pred = np.array([a < b for a, b in prob]).astype(int)


# Calculate the table of prediction result
def cal_result(pred_y, label):
    result = np.zeros([2, 2])
    for p, l in zip(pred_y, label.values):
        result[l[0]][p] += 1
    return result


print(cal_result(pred, label_test))


# Select mis-classified emails in test data
def choose_mis(pred_y, label):
    misidx = []
    for i, p in enumerate(zip(pred_y, label.values)):
        if p[0] != p[1]:
            misidx.append(i)
    return misidx


# Load feature name from readme for plot
names = []
with open('README.md', 'r') as f:
    for l in f:
        names.append(l)
names = [n.replace('\n', '') for n in names[2:] if len(n.replace('\n', '')) > 0]


# Calculate the real probability based on the log probability
def calRealProbability(a, b):
    sum = np.add(np.exp(a), np.exp(b))
    return [np.divide(np.exp(a), sum), np.divide(np.exp(b), sum)]


# Need to mark names and give probability
prob_c = [calRealProbability(a, b) for a, b in prob]
mis_idx = choose_mis(pred, label_test)
mis_idx = [mis_idx[i] for i in np.random.choice(len(mis_idx), 3).astype(int)]
idx = mis_idx
print(x_test.iloc[idx], pred[idx], [[prob[i], prob_c[i]] for i in idx], label_test.iloc[idx].values)
for i, m in enumerate(idx):
    fig = plt.figure(num=None, figsize=(16, 9), dpi=480)
    xx = x_test.iloc[m].values
    plt.title(
        "Features of Misclassified emails classify " + str(label_test.iloc[m].values[0]) + " as " + str(pred[m])+" #"+str(i))
    plt.plot(np.arange(nbc.dim), xx, 'ro', label='email')
    plt.plot(np.arange(nbc.dim), nbc.average_lambda(x_train, label_train, 0), 'go', label='Average Lambda y=0')
    plt.plot(np.arange(nbc.dim), nbc.average_lambda(x_train, label_train, 1), 'bo', label='Average Lambda y=1')
    plt.xticks(np.arange(nbc.dim), names, rotation='vertical')
    plt.xlabel("Index of Features")
    plt.ylabel("Value of Features")
    plt.legend()
    plt.savefig('mis_'+str(i))
    #plt.show()

# Calculate the real probability and pick the three most ambiguity emails.
most_ambi = np.array([np.abs(p[0]-p[1]) for p in prob_c])
most_ambi.sort()
ambi_idx = [i for i, p in enumerate(prob_c) if np.abs(p[0]-p[1]) in most_ambi[:3]]
idx = ambi_idx
print(x_test.iloc[idx], pred[idx], [[prob[i], prob_c[i]] for i in idx], label_test.iloc[idx].values)
for i, m in enumerate(idx):
    fig = plt.figure(num=None, figsize=(16, 9), dpi=480)
    xx = x_test.iloc[m].values
    plt.title(
        "Features of Most Ambiguilty emails " + str(label_test.iloc[m].values[0]) + " classified as " + str(pred[m])+" #"+str(i))
    plt.plot(np.arange(nbc.dim), xx, 'ro', label='email')
    plt.plot(np.arange(nbc.dim), nbc.average_lambda(x_train, label_train, 0), 'go', label='Average Lambda y=0')
    plt.plot(np.arange(nbc.dim), nbc.average_lambda(x_train, label_train, 1), 'bo', label='Average Lambda y=1')
    plt.xticks(np.arange(nbc.dim), names, rotation='vertical')
    plt.xlabel("Index of Features")
    plt.ylabel("Value of Features")
    plt.legend()
    plt.savefig("Ambig_" + str(i))
    #plt.show()

