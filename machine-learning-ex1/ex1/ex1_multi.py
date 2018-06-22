import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    X = []
    Y = []

    with open(file_name, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(',')
            X.append(fields[:-1])
            Y.append(fields[-1])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Y = np.expand_dims(Y, axis=1)

    return X, Y

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, ddof=1, axis=0)

    return (data - mu) / std, mu, std

def compute_cost_multi(X, Y, theta):
    m = X.shape[0]
    pred = np.dot(X, theta)
    return 1.0 / (2 * m) * np.sum(np.square(pred - Y))

def gradient_descent_multi(X, Y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        pred = np.dot(X, theta)
        grad = np.dot(X.T, (pred - Y))
        theta = theta - 1.0/m * alpha * grad

        J_history[i] = compute_cost_multi(X, Y, theta)

    return theta, J_history

def normal_eqn(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


X, Y = load_data("ex1data2.txt")
X , mu, std = feature_normalize(X)
O = np.ones((X.shape[0], 1))
X = np.concatenate((O, X), axis=1)

alpha = 0.01
num_iters = 400

theta = np.zeros((X.shape[1], 1))
theta, J_history = gradient_descent_multi(X, Y, theta, alpha, num_iters)

print theta

'''
x = range(len(J_history))
plt.plot(x, J_history)
plt.show()
'''

predict1 = [1650, 3]
predict1 = np.array(predict1)
norm_predict = (predict1 - mu) / std
o = np.ones((1,1))
norm_predict = np.expand_dims(norm_predict, axis=0)
final_predict = np.concatenate((o, norm_predict), axis=1)
result = final_predict.dot(theta)
print result[0]

X, Y = load_data("ex1data2.txt")
X = np.concatenate((O, X), axis=1)
eq_theta = normal_eqn(X, Y)
print eq_theta.shape

predict2 = [1, 1650, 3]
predict2 = np.expand_dims(np.array(predict2), axis=0)
result2 = predict2.dot(eq_theta)
print result2