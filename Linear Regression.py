import numpy as np


def normal_equation(X, y):
    X = np.array(X)
    y = np.array(y)
    return np.linalg.inv(X.T @ X) @ X.T @ y

def gradient_descent(X, y, lr, num_iterations):
    samples, features = X.shape
    weights = np.zeros(features)
    bias = 0
    for i in range(num_iterations):
        pred = X @ weights.T + bias
        err = np.mean((y - pred) ** 2)
        dw = 2/samples * (pred - y) @ X
        db = 2/samples * sum(pred - y)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias, err

def sgd(X, y, lr, num_iterations):
    samples, features = X.shape
    weights = np.zeros(features)
    bias = 0
    for i in range(num_iterations):
        idx = np.random.randint(0, samples-1)
        pred = np.dot(X[idx], weights) + bias
        err = (y[idx] - pred) ** 2
        dw = 2/samples * (pred - y[idx]) * X[idx]
        db = 2/samples * (pred - y[idx])
        weights -= lr * dw
        bias -= lr * db
    return weights, bias, err

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
print(sgd(X, y, 0.01, 1000))

