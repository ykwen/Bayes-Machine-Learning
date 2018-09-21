import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.naive_bayes import MultinomialNB

label_train = pd.read_csv('label_train.csv', header=None)
label_test = pd.read_csv('label_test.csv', header=None)
x_train = pd.read_csv('x_train.csv', header=None)
x_test = pd.read_csv('x_test.csv', header=None)

clf = MultinomialNB()

clf.fit(x_train, label_train)
print(clf.score(x_test, label_test))

count = np.zeros([2, 2])
for x, y in zip(x_test.values, label_test.values):
    p = clf.predict([x])
    count[y[0]][p[0]] += 1
print(count)