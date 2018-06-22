import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name):
    #load data
    X = []
    Y = []

    with open(file_name, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(',')
            X.append(float(fields[0]))
            Y.append(float(fields[1]))

    X = np.expand_dims(np.array(X, dtype=np.float32), axis=1)
    Y = np.expand_dims(np.array(Y, dtype=np.float32), axis=1)

    O = np.ones((len(X),1))
    X = np.concatenate((O, X), axis=1)
    return X, Y

def compute_cost(X, Y, theta):
    m = int(len(X))
    pred = np.dot(X, theta)
    return 1.0 / (2 * m) * np.sum(np.square(pred - Y))

def gradient_descent(X, Y, theta, alpha, iterations):
    m = int(len(X))
    J_history = np.zeros((iterations,1))

    for i in range(iterations):
        pred = np.dot(X, theta)
        grad_0 = 1.0 / m * alpha * np.sum(pred - Y)
        grad_1 = 1.0 / m * alpha * np.sum((pred - Y) * np.expand_dims(X[:,1], axis=1))
        theta[0] = theta[0] - grad_0
        theta[1] = theta[1] - grad_1

        J_history[i] = compute_cost(X, Y, theta)

    return theta, J_history

def gradient_descent_vec(X, Y, theta, alpha, iterations):
    m = int(len(X))
    J_history = np.zeros((iterations,1))

    for i in range(iterations):
        pred = np.dot(X, theta)
        grad = np.dot(X.T, (pred - Y))
        theta = theta - 1.0 / m * alpha * grad

        J_history[i] = compute_cost(X, Y, theta)

    return theta, J_history

X, Y = load_data('ex1data1.txt')
theta = np.zeros((2, 1))
alpha = 0.01
iterations = 1500

theta, loss = gradient_descent_vec(X, Y, theta, alpha, iterations)

print loss
'''
y_pred = np.dot(X, theta)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], y_pred)
plt.show()
'''