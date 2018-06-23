import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def load_data(file_name):
    X = []
    Y = []

    with open(file_name, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(',')
            X.append(fields[:-1])
            Y.append([fields[-1]])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

    return X, Y

def plot_data(X, Y):
    pos = np.where(Y == 1)
    neg = np.where(Y == 0)

    X_pos = X[pos[0]]
    X_neg = X[neg[0]]

    plt.scatter(X_pos[:,0], X_pos[:,1])
    plt.scatter(X_neg[:,0], X_neg[:,1], marker='x')

    plt.show()

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def cost_function(X, Y, theta):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    #print h
    J = (-(Y.T).dot(np.log(h)) - ((1 - Y).T).dot(np.log(1 - h))) / m
    grad = X.T.dot(h - Y) / m

    return J, grad


X, Y = load_data("ex2data1.txt")
#plot_data(X, Y)

m, n = X.shape
O = np.ones((m, 1))
X = np.concatenate((O, X), axis=1)

initial_theta = np.random.rand(n + 1, 1) * 0.001
J, grad = cost_function(X, Y, initial_theta)

print J, grad

