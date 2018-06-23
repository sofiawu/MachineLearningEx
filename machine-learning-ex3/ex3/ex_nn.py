import numpy as np
import scipy.io as sio


def load_data(file_name):
    data = sio.loadmat(file_name)

    return np.array(data['X'], dtype=np.float32), np.array(data['y'], dtype=np.uint32)

def load_weights(file_name):
    weights = sio.loadmat(file_name)
    return np.array(weights['Theta1'], dtype=np.float32), np.array(weights['Theta2'], dtype=np.float32)

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def predict(X, theta1, theta2):
    print X.shape
    O = np.ones((X.shape[0], 1))
    X = np.concatenate((O, X), axis=1)

    a1 = sigmoid(X.dot(theta1.T))
    a1 = np.concatenate((O, a1), axis=1)
    a2 = sigmoid(a1.dot(theta2.T))

    pred = np.argmax(a2, axis=1)
    return pred + 1


X, y = load_data('ex3data1.mat')
y = y.squeeze(axis=1)

theta1, theta2 = load_weights('ex3weights.mat')

pred = predict(X, theta1, theta2)

acc = np.mean(pred == y) * 100.

print acc

print predict(np.expand_dims(X[0], axis=0), theta1, theta2)




