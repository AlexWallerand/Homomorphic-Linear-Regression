import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from homomorphic import *

def init_theta(data):
    return np.zeros((data.shape[1], 1))

def model(X, theta):
    return test_matrice_mul(X, theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X ,y, theta):
    m = len(y)
    return test_matrice_scalar_mul(test_matrice_mul(test_matrice_transpose(X), test_matrice_sou(model(X, theta) , y)), 1/m)

def gradient_descent(X, y, theta, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

def regression(X, y):
    theta = init_theta(X)
    theta_f = gradient_descent(X, y, theta, learning_rate=0.1, n_iteration=1000)
    return model(X, theta_f)

def load_data(path, column1, column2):
    data = pd.read_csv(path)
    x, y = data[column1].values, data[column2].values
    X = np.column_stack((x,np.ones(x.shape)))
    y = y.reshape(y.shape[0], 1)
    return x, X, y


#~~~~~~~~~~~~~~~~~~~~ Programme Principal ~~~~~~~~~~~~~~~~~~~~

x, X, y = load_data('Car_sales.csv','Engine_size','Horsepower')
regression = regression(X, y)

plt.scatter(x, y)
plt.plot(x, regression, color='green')
plt.show()