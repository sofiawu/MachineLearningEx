import numpy as np
import math

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

def map_feature(X):
    degree = 7

    out = np.zeros((X.shape[0], ))
    print out.shape
    for i in range(degree):
        for j in range(i + 1):
            col = (np.power(X[:,0], (i - j)) * np.power(X[:,1], (j)))
            out = np.column_stack((out, col))
    out = np.delete(out, 0, axis=1)
    return out

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def cost_function_reg(X, Y, theta, lamba):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    J = (-(Y.T).dot(np.log(h)) - ((1 - Y).T).dot(np.log(1 - h)) + lamba / 2. * (theta.T).dot(theta)) / m
    grad = (X.T).dot(h - Y) / m + lamba * theta / m
    return J, grad

X, Y = load_data('ex2data2.txt')
X = map_feature(X)
print X.shape

initial_theta = np.zeros((X.shape[1], 1))
lamba = 1

J, grad = cost_function_reg(X, Y, initial_theta, lamba)
print J, grad