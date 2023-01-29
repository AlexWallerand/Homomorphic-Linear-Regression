import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from homomorphic import *
from Pyfhel import Pyfhel

def init_theta(data):
    return np.zeros((data.shape[1], 1))

def model(X, theta):
    return matrice_mul(X, theta, HE)

def grad(X ,y, theta):
    m = len(y)
    ctxt_m = HE.encryptFrac(np.array([1/m], dtype=np.float64))
    return matrice_scalar_mul(matrice_mul(matrice_transpose(X, HE), matrice_sou(model(X, theta) , y, HE), HE), ctxt_m, HE)

def gradient_descent(X, y, theta, learning_rate, n_iteration):
    ctxt_learning_rate = HE.encryptFrac(np.array([learning_rate], dtype=np.float64))
    for i in range(0, n_iteration):
        print(i)
        theta = matrice_sou(theta, matrice_scalar_mul(grad(X, y, theta), ctxt_learning_rate, HE), HE)
    return theta

def regression(X, y):
    theta = init_theta(X)
    ctxt_theta = matrice_encrypt(theta, HE)
    theta_f = gradient_descent(X, y, ctxt_theta, learning_rate=0.1, n_iteration=2)
    return model(X, theta_f)

def load_data(path, column1, column2):
    data = pd.read_csv(path)
    x, y = data[column1].values, data[column2].values
    X = np.column_stack((x,np.ones(x.shape)))
    y = y.reshape(y.shape[0], 1)
    return x, X, y


#~~~~~~~~~~~~~~~~~~~~ Programme Principal ~~~~~~~~~~~~~~~~~~~~

# Initialisation du modèle d'encryption Pyfhel
HE = Pyfhel()
qi_sizes = [60] + [30]*10 + [60]
HE.contextGen(scheme="ckks", n=2**14, scale=2**30, qi_sizes=qi_sizes)
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()

# Récupération des données
x, X, y = load_data('Car_sales.csv','Engine_size','Horsepower')
ptxt_x = np.array(X[:10], dtype=np.float64)
ptxt_y = np.array(y[:10], dtype=np.float64)

# Encryption des données
ctxt_x = matrice_encrypt(ptxt_x, HE)
ctxt_y = matrice_encrypt(ptxt_y, HE)

# Calcul de la régression cryptée
regression = regression(ctxt_x, ctxt_y)
result = decrypt(regression, HE)
print(result)

"""plt.scatter(x, y)
plt.plot(x[:10], result, color='green')
plt.savefig("result.png")"""