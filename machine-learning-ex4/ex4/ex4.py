import numpy as np
import scipy.io as sio

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

def load_data(file_name):
    data = sio.loadmat(file_name)
    X = np.array(data['X'], dtype=np.float32)
    y = np.zeros((X.shape[0], num_labels), dtype=np.float32)

    for i in range(num_labels):
        y[:, i] = ((np.array(data['y']) - 1).squeeze() == i)

    return X, y, np.array(data['y'], dtype=np.int32).squeeze()

def load_weights(file_name):
    weights = sio.loadmat(file_name)

    return np.array(weights['Theta1'], dtype=np.float32), np.array(weights['Theta2'], dtype=np.float32)

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * ( 1 - sigmoid(x))

def nn_cost_function(X, y, theta1, theta2, lamba):
    m = X.shape[0]

    #forward
    O = np.ones((m, 1))
    X = np.concatenate((O, X), axis=1)
    z1 = X.dot(theta1.T)
    a1 = sigmoid(z1)
    a1 = np.concatenate((O, a1), axis=1)
    h = sigmoid(a1.dot(theta2.T))

    h_flatten = h.reshape(-1,1)
    y_flatten = y.reshape(-1,1)
    theta1_flatten = theta1[:, 1:].reshape(-1,1)
    theta2_flatten = theta2[:, 1:].reshape(-1,1)

    J = -(y_flatten.T.dot(np.log(h_flatten)) + (1 - y_flatten).T.dot(np.log(1 - h_flatten))) / m  + lamba / (2.0 * m) * (theta1_flatten.T.dot(theta1_flatten) + theta2_flatten.T.dot(theta2_flatten))

    b2 = h - y
    theta2_grad = b2.T.dot(a1) + lamba * theta2
    b1 = (b2.dot(theta2[:,1:])) * sigmoid_gradient(z1)
    theta1_grad = b1.T.dot(X) + lamba * theta1

    return J, [theta1_grad, theta2_grad]

def rand_initialize_weights(L_in, L_out):
    eps = np.power((6./L_in), 0.5)
    return np.random.rand(L_out, L_in + 1) * 2 * eps - eps

def gradient_descent(X, Y, theta, alpha, lamba, num_iters):
    m = X.shape[0]
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        J, theta_grad = nn_cost_function(X, Y, theta[0], theta[1], lamba)
        theta[0] = theta[0] - alpha * theta_grad[0]
        theta[1] = theta[1] - alpha * theta_grad[1]

        J_history[i] = J

    return theta, J_history


def predict(X, theta1, theta2):
    m = X.shape[0]

    # forward
    O = np.ones((m, 1))
    X = np.concatenate((O, X), axis=1)
    z1 = X.dot(theta1.T)
    a1 = sigmoid(z1)
    a1 = np.concatenate((O, a1), axis=1)
    h = sigmoid(a1.dot(theta2.T))

    pred = np.argmax(h, axis=1)
    return pred + 1


X, y, y_label = load_data("ex4data1.mat")
#print X.shape, y.shape

theta1, theta2 = load_weights("ex4weights.mat")
#print theta1.shape, theta2.shape

lamba = 1

J, (theta1_grad, theta2_grad) = nn_cost_function(X, y, theta1, theta2, lamba)
print J

#print sigmoid_gradient(np.array([1,-0.5, 0, 0.5, 1]))

initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

theta, loss = gradient_descent(X, y, [initial_theta1, initial_theta2], 0.0001, lamba, 2000)

print loss[-1]
pred = predict(X, theta[0], theta[1])

acc = np.mean(pred == y_label) * 100.

print acc