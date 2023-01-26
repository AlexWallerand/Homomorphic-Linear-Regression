import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linear_reg_without_numpy as lrwnp


def init_theta(data):
    return np.zeros((data.shape[1], 1))


def model(X, theta):
    return np.dot(X, theta)


def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta


def load_data(path, column1, column2):
    data = pd.read_csv(path)
    x, y = data[column1].values, data[column2].values
    X = np.column_stack((x, np.ones(x.shape)))
    y = y.reshape(y.shape[0], 1)
    return x, X, y

# ~~~~~~~~~~~~~~~~~~~~ Programme Principal ~~~~~~~~~~~~~~~~~~~~


x, X, y = load_data('Car_sales.csv', 'Engine_size', 'Horsepower')
theta = init_theta(X)

theta_f = gradient_descent(X, y, theta, learning_rate=0.01, n_iteration=10000)
model_regression = model(X, theta_f)

theta_f_without_np = lrwnp.gradient_descent(
    X, y, theta, learning_rate=0.01, n_iteration=10000)
model_regression_without_np = lrwnp.model(X, theta_f_without_np)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Calculs avec Numpy (vert) et sans Numpy (rouge)')
ax1.scatter(x, y)
ax1.plot(x, model_regression, color='green')
ax1.text(2, 430, 'learning_rate = 0.01  n_iteration = 10000',
         fontsize=10, bbox=dict(facecolor='green', alpha=0.3))
ax2.scatter(x, y)
ax2.plot(x, model_regression_without_np, color='red')
ax2.text(2, 430, 'learning_rate = 0.01  n_iteration = 10000',
         fontsize=10, bbox=dict(facecolor='red', alpha=0.3))

#plt.scatter(x, y)
#plt.xlabel('Engine size')
#plt.ylabel('Horse power')
#plt.plot(x, model_regression, color='green')
#plt.text(2, 430, 'learning_rate = 0.01  n_iteration = 10000', fontsize=10, bbox=dict(facecolor='green', alpha=0.3))
plt.show()
